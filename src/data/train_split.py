import numpy as np
import pandas as pd
import argparse


def train_slit(prod_info, split_size=100000):
    n = prod_info.shape[0]
    idx = np.arange(n)
    np.random.seed(321)
    np.random.shuffle(idx)
    split = np.zeros(n, dtype=np.bool)
    split[idx[:-split_size]] = True
    return prod_info.assign(train=split)[['product_id', 'train']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod_info_csv', required=True,
                        help='Path to training prod info csv')
    parser.add_argument('--output_file', required=True,
                        help='Path to save split into')
    args = parser.parse_args()
    prod_info = pd.read_csv(args.prod_info_csv)
    split = train_slit(prod_info)
    split.to_csv(args.output_file, index=False)
