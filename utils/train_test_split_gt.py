import argparse

import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split train and test gt')
    parser.add_argument('--gt_csv', type=str, default='/mldisk/nfs_shared_/MLVD/DISC/dev_ground_truth.csv')
    parser.add_argument('--save_name', type=str, default='disc')
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt_csv, names=['q', 'r'])
    gt_df['q'] = 'dev_queries/' + gt_df['q'] + '.jpg'
    gt_df['r'] = 'references/' + gt_df['r'] + '.jpg'
    # import pdb; pdb.set_trace()
    shuffled_gt_df = np.random.permutation(gt_df)
    pd.DataFrame(shuffled_gt_df[:-1000]).to_csv(f'{args.save_name}_train.csv', index=False)
    pd.DataFrame(shuffled_gt_df[-1000:]).to_csv(f'{args.save_name}_test.csv', index=False)

    # import pdb; pdb.set_trace()
