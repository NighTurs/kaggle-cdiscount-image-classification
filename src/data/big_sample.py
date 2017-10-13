import numpy as np
import pandas as pd
import argparse

def create_big_sample(prod_info_csv):
    prod_info = pd.read_csv(prod_info_csv)
    category_stats = prod_info.groupby(by='category_id').size()
    category_stats.sort_values(ascending=False, inplace=True)
    categories = category_stats[:2000].index.values
    categories_set = set(categories)

    np.random.seed(123)
    chunks = []
    for category, prods in prod_info.groupby(by='category_id'):
        if not category in categories_set:
            continue
        chunks.append(prods.sample(100 if prods.shape[0] >= 100 else prods.shape[0]))
    sample = pd.concat(chunks)

    sample = sample.reset_index(drop=True)
    idx = np.arange(sample.shape[0])
    np.random.seed(123)
    np.random.shuffle(idx)
    cut = int(sample.shape[0] * 0.8)
    sample['train'] = False
    sample.loc[idx[:cut], 'train'] = True

    return sample


def save_big_sample(big_sample, output_file):
    big_sample.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod_info_csv', required=True, help='Path to prod info csv')
    parser.add_argument('--output_file', required=True, help='File to save sample into')

    args = parser.parse_args()
    big_sample = create_big_sample(args.prod_info_csv)
    save_big_sample(big_sample, args.output_file)
