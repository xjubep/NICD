import argparse
import os

import cv2
import faiss
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import torch.linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.disc import DISC
from datasets.fivr import FIVR
from datasets.simulated import SIMULATED
from isc.descriptor_matching import match_and_make_predictions, knn_match_and_make_predictions
from isc.io import read_ground_truth
from isc.metrics import evaluate, print_metrics
from utils import *


def unsqueeze_path(path):
    return path[0]


def fivr_path_to_id(path):
    return ''.join(path.split('/')[-3:])[:-4]


def disc_path_to_id(path):
    return ''.join(path.split('/')[-1:])[:-4]


def plot_and_save(args, idx, query_paths, reference_paths, I, root_dir):
    query_path = os.path.join(root_dir, query_paths[idx][0])
    reference_path = os.path.join(root_dir, reference_paths[idx][0])
    retrieved_paths = [os.path.join(root_dir, reference_paths[x][0]) for x in I[idx]]

    if args.dataset == 'FIVR':
        query_id = fivr_path_to_id(query_path)
        reference_id = fivr_path_to_id(reference_path)
        retrieved_ids = list(map(fivr_path_to_id, retrieved_paths))
    elif args.dataset == 'DISC':
        query_id = disc_path_to_id(query_path)
        reference_id = disc_path_to_id(reference_path)
        retrieved_ids = list(map(disc_path_to_id, retrieved_paths))

    fig = plt.figure(figsize=(10, 6))
    img = mpimg.imread(query_path)
    img = cv2.resize(img, (224, 224))
    fig.add_subplot(3, 5, 5 * 0 + 1)
    plt.title(f'{query_id}')

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    img = mpimg.imread(reference_path)
    img = cv2.resize(img, (224, 224))
    fig.add_subplot(3, 5, 5 * 0 + 2)
    plt.title(f'{reference_id}')

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    for j in range(10):
        img = mpimg.imread(retrieved_paths[j])
        img = cv2.resize(img, (224, 224))
        fig.add_subplot(3, 5, 5 * 1 + j + 1)
        plt.title(f'{retrieved_ids[j]}')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

    save_fig_dir = f'/mldisk/nfs_shared_/sy/{args.dataset}/retrieval_samples/{args.model}_{args.train_mode}_rs{args.ref_size}_is{args.img_size}'

    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)

    plt.savefig(f'{save_fig_dir}/GT{idx:03d}_{query_id}_{reference_id}.png')
    plt.close(fig)


@torch.no_grad()
def extract_feat(args, net, loader, save_feat_dir):
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
    np.save(save_feat_dir, feats)
    return feats


def get_avg_rank_and_reall(args, net, query_loader, ref_loader, gt_csv, query_csv, ref_csv, root_dir, query_dataset,
                           ref_dataset):
    query_paths = pd.read_csv(query_csv).values
    reference_paths = pd.read_csv(ref_csv).values

    save_feat_dir = f'/mldisk/nfs_shared_/sy/{args.dataset}/features_new_fivr/{args.model}_{args.train_mode}_is{args.img_size}'
    if not os.path.exists(save_feat_dir):
        os.makedirs(save_feat_dir)

    save_query_feat_dir = os.path.join(save_feat_dir, f'feats_query_{args.query_size}')
    save_ref_feat_dir = os.path.join(save_feat_dir, f'feats_ref_{args.ref_size}')

    if os.path.exists(f'{save_query_feat_dir}.npy'):
        feats_query = np.load(f'{save_query_feat_dir}.npy')
    else:
        feats_query = extract_feat(args, net, query_loader, save_query_feat_dir)

    if os.path.exists(f'{save_ref_feat_dir}.npy'):
        feats_ref = np.load(f'{save_ref_feat_dir}.npy')
    else:
        feats_ref = extract_feat(args, net, ref_loader, save_ref_feat_dir)

    index = faiss.IndexFlatIP(feats_ref.shape[1])
    index.add(feats_ref)

    D, I = index.search(feats_query, feats_ref.shape[0])

    gt_csv = pd.read_csv(gt_csv, header=None).values
    # import pdb; pdb.set_trace()
    ranks, dists = [], []
    bar = tqdm(enumerate(gt_csv), 'Plotting', total=len(gt_csv), ncols=120)
    for idx, gt in bar:
        if args.visualize:
            plot_and_save(args, idx, query_paths, reference_paths, I, root_dir)
        ranks.append(np.where(I[idx] == idx)[0][0])
        dists.append(abs(1 - D[idx][ranks[-1]]))

    avg_rank = sum(ranks) / len(ranks)
    avg_dist = sum(dists) / len(dists)
    # plt.title(f'{args.model.upper()} Dist{avg_dist:.3f}')
    # plt.hist(dists, bins=100)
    # plt.show()
    # import pdb; pdb.set_trace()
    print(f'Rank/Frame: {avg_rank:.1f}, Distance/Frame: {avg_dist:.3f}')

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
    print(f"R@1 {recall.loc[0]['recall']:.3f} | R@5 {recall.loc[4]['recall']:.3f} | R@10 {recall.loc[9]['recall']:.3f}")
    return


def get_micro_ap(args, net, query_loader, ref_loader, gt_csv, query_csv, ref_csv, query_dataset, ref_dataset):
    gt = read_ground_truth(gt_csv)
    query_paths = pd.read_csv(query_csv).values
    reference_paths = pd.read_csv(ref_csv).values

    query_paths = list(map(unsqueeze_path, query_paths))
    reference_paths = list(map(unsqueeze_path, reference_paths))

    query_ids = list(map(disc_path_to_id, query_paths))
    reference_ids = list(map(disc_path_to_id, reference_paths))

    save_feat_dir = f'/mldisk/nfs_shared_/sy/{args.dataset}/features_new/{args.model}_{args.train_mode}_is{args.img_size}'
    save_query_feat_dir = os.path.join(save_feat_dir, f'feats_query_{args.query_size}')
    save_ref_feat_dir = os.path.join(save_feat_dir, f'feats_ref_{args.ref_size}')
    save_train_feat_dir = os.path.join(save_feat_dir, f'feats_train_1m')

    if os.path.exists(f'{save_query_feat_dir}.npy'):
        feats_query = np.load(f'{save_query_feat_dir}.npy')
    else:
        feats_query = extract_feat(args, net, query_loader, save_query_feat_dir)

    if os.path.exists(f'{save_ref_feat_dir}.npy'):
        feats_ref = np.load(f'{save_ref_feat_dir}.npy')
    else:
        feats_ref = extract_feat(args, net, ref_loader, save_ref_feat_dir)

    # if os.path.exists(f'{save_train_feat_dir}.npy'):
    #     feats_train = np.load(f'{save_train_feat_dir}.npy')
    # else:
    #     feats_train = extract_feat(args, net, train_loader, save_train_feat_dir)

    d = feats_ref.shape[1]

    #### MAKE PREDICTION ####
    if args.knn == -1:
        print(
            f"Track 2 running matching of {len(query_ids)} queries in {len(reference_ids)} database ({d}D descriptors), "
            f"max_results={args.max_results}."
        )
        predictions = match_and_make_predictions(
            feats_query, query_ids,
            feats_ref, reference_ids,
            args.max_results,
            metric=faiss.METRIC_INNER_PRODUCT if args.ip else faiss.METRIC_L2
        )
    else:
        print(
            f"Track 2 running matching of {len(query_ids)} queries in {len(reference_ids)} database ({d}D descriptors), "
            f"kNN with k={args.knn}."
        )
        predictions = knn_match_and_make_predictions(
            feats_query, query_ids,
            feats_ref, reference_ids,
            args.knn,
            metric=faiss.METRIC_INNER_PRODUCT if args.ip else faiss.METRIC_L2
        )
    print('> PREDICTION MADE')
    ########################

    #### EVALUATE #####
    print(f"Evaluating {len(predictions)} predictions ({len(gt)} GT matches)")
    metrics = evaluate(gt, predictions)
    print_metrics(metrics)
    # print(f'{args.model.upper()} (ref_size={args.ref_size} | k={args.knn:2d}) AP: {metrics.average_precision:.5f}')
    print('> EVALUATED')
    ###################
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='SIMULATED', choices=['FIVR', 'DISC', 'SIMULATED'])
    parser.add_argument('-sd', '--save_dir', type=str, default='/mldisk/nfs_shared_/sy/DISC/results')
    parser.add_argument('-sfd', '--save_fig_dir', type=str, default='/mldisk/nfs_shared_/sy/DISC/retrieval_samples')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        default='/mldisk/nfs_shared_/sy/DISC/checkpoints/hybrid_vit_1129_061226/hybrid_vit_epoch07_ckpt.pth')

    parser.add_argument('-m', '--model', type=str, default='mobilenet',
                        choices=['mobilenet', 'efficientnet', 'hybrid_vit', 'mobilenet_dolg', 'mobilenet_dolg_df',
                                 'mobilenet_dolg_ff'])
    parser.add_argument('-is', '--img_size', type=int, default=224)
    parser.add_argument('-se', '--seed', type=int, default=42)

    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-nw', '--num_workers', type=int, default=8)

    parser.add_argument('-tm', '--train_mode', type=str, default='online', choices=['triplet', 'online'])
    parser.add_argument('-qs', '--query_size', type=str, default='1k', choices=['1k', '5k', '50k'])
    parser.add_argument('-rs', '--ref_size', type=str, default='20k', choices=['1k', '20k', '100k', '1m'])

    parser.add_argument('-mr', '--max_results', type=int, default=50_000)
    parser.add_argument('-k', '--knn', type=int, default=1)
    parser.add_argument('--ip', type=bool, default=True)

    # amp config:
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # print(args)
    #### SEED EVERYTHING ####
    seed_everything(args.seed)
    #########################

    #### BUILD DATASET ####
    query_dataset, ref_dataset, train_dataset = None, None, None

    query_csv = f'/workspace/gt/{args.dataset.lower()}_query_{args.query_size}.csv'
    ref_csv = f'/workspace/gt/{args.dataset.lower()}_ref_{args.ref_size}.csv'
    # train_csv = f'/workspace/gt/{args.dataset.lower()}_train_1m.csv'
    gt_csv = f'/workspace/gt/{args.dataset.lower()}_test_gt.csv'
    # gt_csv = f'/workspace/gt/final_ground_truth.csv'

    if args.dataset == 'FIVR':
        root_dir = '/mldisk/nfs_shared_/MLVD/FIVR/frames'
        query_dataset = FIVR(args, img_root=root_dir, query_csv=query_csv, mode='query')
        ref_dataset = FIVR(args, img_root=root_dir, ref_csv=ref_csv, mode='ref')
    elif args.dataset == 'DISC':
        root_dir = '/mldisk/nfs_shared_/MLVD/DISC/images'
        query_dataset = DISC(args, img_root=root_dir, query_csv=query_csv, mode='query')
        ref_dataset = DISC(args, img_root=root_dir, ref_csv=ref_csv, mode='ref')
        # train_dataset = DISC(args, img_root=root_dir, ref_csv=train_csv, mode='ref')
    elif args.dataset == 'SIMULATED':
        root_dir = None
        query_dataset = SIMULATED(args, query_csv=query_csv, mode='query')
        ref_dataset = SIMULATED(args, ref_csv=ref_csv, mode='ref')

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print('> DATAMODULE BUILT')
    ######################

    #### BUILD MODEL ####
    triple_model = build_model(args)
    print(f'> {args.model.upper()} MODEL BUILT')
    #####################

    #### LOAD CHECKPOINT ####
    state_dict = torch.load(args.checkpoint)['state_dict']
    triple_model.module.embedding_net.load_state_dict(state_dict)
    print('> CHECKPOINT LOADED')
    #########################

    get_avg_rank_and_reall(args, triple_model, query_loader, ref_loader, gt_csv, query_csv, ref_csv, root_dir,
                           query_dataset, ref_dataset)
    if args.dataset == 'DISC':
        get_micro_ap(args, triple_model, query_loader, ref_loader, gt_csv, query_csv, ref_csv, query_dataset,
                     ref_dataset)
