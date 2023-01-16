import argparse
import os
import time

import faiss
import numpy as np
import torch

from utils import build_model


def get_search_time(args, ):
    save_feat_dir = f'/mldisk/nfs_shared_/sy/{args.dataset}/features_new/{args.model}_{args.train_mode}_is{args.img_size}'

    save_query_feat_dir = os.path.join(save_feat_dir, f'feats_query_{args.query_size}')
    save_ref_feat_dir = os.path.join(save_feat_dir, f'feats_ref_{args.ref_size}')
    feats_query = np.load(f'{save_query_feat_dir}.npy')
    feats_ref = np.load(f'{save_ref_feat_dir}.npy')

    index = faiss.IndexFlatIP(feats_ref.shape[1])
    index.add(feats_ref)

    start = time.time()
    D, I = index.search(feats_query, feats_ref.shape[0])
    end = time.time()
    return (end - start) * 1000.0 / feats_query.shape[0]


def get_avg_search_time(args):
    repetitions = 10
    timings = np.zeros((repetitions, 1))
    for rep in range(repetitions):
        curr_time = get_search_time(args)
        timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f'Mean Search time: {mean_syn:.3f} ms, Std: {std_syn:.3f} ms')


def get_descriptor_time(args):
    triple_model = build_model(args)
    triple_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(args.device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = triple_model(dummy_input, single=True)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = triple_model(dummy_input, single=True)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f'Mean: {mean_syn:.2f} ms, Std: {std_syn:.2f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='DISC', choices=['FIVR', 'DISC'])
    parser.add_argument('-m', '--model', type=str, default='mobilenet',
                        choices=['mobilenet', 'efficientnet', 'hybrid_vit', 'mobilenet_dolg', 'mobilenet_dolg_df',
                                 'mobilenet_dolg_ff'])
    parser.add_argument('-is', '--img_size', type=int, default=224)

    parser.add_argument('-tm', '--train_mode', type=str, default='online', choices=['triplet', 'online'])
    parser.add_argument('-qs', '--query_size', type=str, default='1k', choices=['1k', '5k'])
    parser.add_argument('-rs', '--ref_size', type=str, default='100k', choices=['1k', '20k', '100k'])

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)

    get_avg_search_time(args)
