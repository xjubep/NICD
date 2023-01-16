import argparse
import os
from datetime import datetime

import faiss
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.disc import DISC
from datasets.fivr import FIVR
from datasets.simulated import SIMULATED
from utils import *


@torch.no_grad()
def extract_feat(args, net, loader):
    net.eval()
    feats = []
    batch_iter = tqdm(enumerate(loader), 'Extracting', total=len(loader), ncols=120)
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        img_flip = batch_item['img_flip'].to(args.device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.model == 'mobilenet_dolg_df':
                feat = torch.cat((net(img, single=True), net(img_flip, single=True)), dim=1).cpu()
            else:
                feat = net(img, single=True).cpu()
        feats.append(feat)
    feats = np.vstack(feats)
    feats = np.float32(feats)
    faiss.normalize_L2(feats)  # 이미 normalize 되어 있어서 필요없긴함
    return feats


def get_avg_rank_and_reall(args, net, query_loader, ref_loader, gt_csv, epoch, wandb):
    feats_query = extract_feat(args, net, query_loader)
    feats_ref = extract_feat(args, net, ref_loader)

    index = faiss.IndexFlatIP(feats_ref.shape[1])
    index.add(feats_ref)

    D, I = index.search(feats_query, feats_ref.shape[0])

    gt_csv = pd.read_csv(gt_csv, header=None).values
    ranks, dists = [], []
    bar = tqdm(enumerate(gt_csv), 'Plotting', total=len(gt_csv), ncols=120)
    for idx, gt in bar:
        try:
            ranks.append(np.where(I[idx] == idx)[0][0])
            dists.append(D[idx][ranks[-1]])
        except IndexError:
            pass

    avg_rank = sum(ranks) / len(ranks)
    avg_dist = sum(dists) / len(dists)
    avg_dist = 1 - avg_dist  # cosine distance
    # print(f'Rank/Frame: {avg_rank:4f}, Distance/Frame: {avg_dist:4f}')

    recall = []
    ranks = np.array(ranks)
    total = len(gt_csv)
    max = np.max(ranks)
    # print(max)

    for a in range(1, 10):
        r = np.where(ranks < a)[0].shape[0]
        recall.append({'topk': a, 'recall': r / total, 'count': r})

    for a in range(10, 1000, 10):
        r = np.where(ranks < a)[0].shape[0]
        recall.append({'topk': a, 'recall': r / total, 'count': r})
    recall = pd.DataFrame(recall)
    top1_recall = recall.loc[0]['recall']
    top5_recall = recall.loc[4]['recall']

    print(
        f'[Epoch {epoch}] Rank: {avg_rank:.4f}, Distance: {avg_dist:.4f}, Top1: {top1_recall:.3f}, Top5: {top5_recall:.3f}')

    if args.wandb:
        wandb.log({'test/rank_per_frame': avg_rank, 'test/distance_per_frame': avg_dist, 'test/recall@1': top1_recall,
                   'test/recall@5': top5_recall},
                  step=epoch)

    del feats_query, feats_ref

    return avg_rank, avg_dist, top1_recall


def train(args, net, loader, optimizer, criterion, warmup_scheduler, scheduler, scaler, epoch, wandb):
    net.train()
    losses = AverageMeter('Loss', ':.4f')

    batch_iter = tqdm(enumerate(loader), 'Training', total=len(loader), ncols=120)
    for batch_idx, batch_item in batch_iter:
        optimizer.zero_grad()
        anc = batch_item['anc'].to(args.device)
        pos = batch_item['pos'].to(args.device)
        neg = batch_item['neg'].to(args.device)

        if epoch <= args.warm_epoch:
            warmup_scheduler.step()

        with torch.cuda.amp.autocast(enabled=args.amp):
            feat1, feat2, feat3 = net(anc, pos, neg)

        loss = criterion(feat1, feat2, feat3)
        losses.update(loss.item())

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler == 'cycle':
            scheduler.step()

        batch_iter.set_description(f'[Epoch {epoch}] '
                                   f'lr: {optimizer.param_groups[0]["lr"]:4f}, '
                                   f'loss: {losses.val:.4f}({losses.avg:.4f}), ')
        batch_iter.update()

    batch_iter.close()

    if args.wandb:
        wandb.log({'train/loss': losses.avg, 'train/learning_rate': optimizer.param_groups[0]["lr"]}, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='SIMULATED', choices=['FIVR', 'DISC', 'SIMULATED'])
    parser.add_argument('-sd', '--save_dir', type=str, default='/mldisk/nfs_shared_/sy/SIMULATED/checkpoints')
    parser.add_argument('-m', '--model', type=str, default='mobilenet',
                        choices=['mobilenet', 'efficientnet', 'hybrid_vit', 'mobilenet_dolg', 'mobilenet_dolg_df',
                                 'mobilenet_dolg_ff'])
    parser.add_argument('-is', '--img_size', type=int, default=224)
    parser.add_argument('-se', '--seed', type=int, default=42)

    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-we', '--warm_epoch', type=int, default=2)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-nw', '--num_workers', type=int, default=8)

    parser.add_argument('-dis', '--distance', type=str, default='cos', choices=['l2', 'cos'])
    parser.add_argument('-mg', '--margin', type=float, default=0.1)

    parser.add_argument('-ot', '--optimizer', type=str, default='adamw',
                        choices=['adam', 'radam', 'adamw', 'adamp', 'ranger', 'lamb', 'adabound'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)

    parser.add_argument('-sc', '--scheduler', type=str, default='cos_base', choices=['cos_base', 'cos', 'cycle'])
    parser.add_argument('-mxlr', '--max_lr', type=float, default=3e-3)  # scheduler - cycle
    parser.add_argument('-mnlr', '--min_lr', type=float, default=1e-6)  # scheduler - cos
    parser.add_argument('-tm', '--tmax', type=float, default=20)  # scheduler - cos
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)

    parser.add_argument('-qs', '--query_size', type=str, default='1k', choices=['1k', '5k'])
    parser.add_argument('-rs', '--ref_size', type=str, default='20k', choices=['20k', '100k'])

    # wandb config:
    parser.add_argument('--wandb', type=bool, default=True)

    # amp config:
    parser.add_argument('--amp', type=bool, default=True)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SEED EVERYTHING ####
    seed_everything(args.seed)
    #########################

    c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')
    save_dir = os.path.join(args.save_dir, f'{args.model}_{c_date}_{c_time}')
    os.makedirs(save_dir)

    #### SET WANDB ####
    run = None
    if args.wandb:
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        wandb.login(key=wandb_api_key)
        run = wandb.init(project='nicd', name=f'{args.model}_{c_date}_{c_time}')
        wandb.config.update(args)
    ###################

    #### BUILD DATASET ####
    train_dataset, test_dataset, query_dataset, ref_dataset = None, None, None, None

    triplet_csv = f'/workspace/triplets/{args.dataset.lower()}_triplet_cos01_50k.csv'
    query_csv = f'/workspace/gt/{args.dataset.lower()}_query_{args.query_size}.csv'
    ref_csv = f'/workspace/gt/{args.dataset.lower()}_ref_{args.ref_size}.csv'
    gt_csv = f'/workspace/gt/{args.dataset.lower()}_test_gt.csv'

    if args.dataset == 'FIVR':
        root_dir = '/mldisk/nfs_shared_/MLVD/FIVR/frames'
        train_dataset = FIVR(args, img_root=root_dir, triplet_csv=triplet_csv, mode='train_triplet')
        query_dataset = FIVR(args, img_root=root_dir, query_csv=query_csv, mode='query')
        ref_dataset = FIVR(args, img_root=root_dir, ref_csv=ref_csv, mode='ref')
    elif args.dataset == 'DISC':
        root_dir = '/mldisk/nfs_shared_/MLVD/DISC/images'
        train_dataset = DISC(args, img_root=root_dir, triplet_csv=triplet_csv, mode='train_triplet')
        query_dataset = DISC(args, img_root=root_dir, query_csv=query_csv, mode='query')
        ref_dataset = DISC(args, img_root=root_dir, ref_csv=ref_csv, mode='ref')
    elif args.dataset == 'SIMULATED':
        root_dir = None
        train_dataset = SIMULATED(args, triplet_csv=triplet_csv, mode='train_triplet')
        query_dataset = SIMULATED(args, query_csv=query_csv, mode='query')
        ref_dataset = SIMULATED(args, ref_csv=ref_csv, mode='ref')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False)
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    iter_per_epoch = len(train_loader)
    print('> DATAMODULE BUILT')
    ######################

    #### BUILD MODEL ####
    triple_model = build_model(args)
    print('> MODEL BUILT')
    ####################

    #### BUILD TRAINER ####
    optimizer = build_optimizer(args, triple_model)
    criterion = build_loss(args)
    scheduler = build_scheduler(args, optimizer, iter_per_epoch)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm_epoch) if args.warm_epoch else None
    print('> TRAINER BUILT')
    #####################

    min_avg_rank, min_avg_dist, max_recall_1 = get_avg_rank_and_reall(args, triple_model, query_loader, ref_loader,
                                                                      gt_csv, 0, wandb)

    print('> START TRAINING')
    for epoch in range(1, args.epochs + 1):
        train(args, triple_model, train_loader, optimizer, criterion, warmup_scheduler, scheduler, scaler, epoch, wandb)
        avg_rank, avg_dist, recall_1 = get_avg_rank_and_reall(args, triple_model, query_loader, ref_loader,
                                                              gt_csv, epoch, wandb)

        if args.scheduler in ['cos_base', 'cos']:
            scheduler.step()

        if max_recall_1 < recall_1:
            max_recall_1 = max(max_recall_1, recall_1)
            torch.save({'epoch': epoch,
                        'state_dict': triple_model.module.embedding_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(save_dir, f'{args.model}_epoch{epoch:02d}_ckpt.pth'))
            print(f'> [SAVE] Epoch {epoch:02d}')

    if args.wandb:
        run.finish()
