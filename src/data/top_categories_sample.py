import pandas as pd
import argparse


def top_categories_sample(prod_info, num_categories):
    category_stats = prod_info.groupby(by='category_id').size()
    category_stats.sort_values(ascending=False, inplace=True)
    categories = category_stats[:num_categories].index.values
    categories_set = set(categories)
    chunks = []
    for category, prods in prod_info.groupby(by='category_id'):
        if category not in categories_set:
            continue
        chunks.append(prods)
    sample = pd.concat(chunks)
    sample = sample.reset_index(drop=True)
    return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod_info_csv', required=True, help='Path to prod info csv')
    parser.add_argument('--output_file', required=True, help='File to save sample into')
    parser.add_argument('--num_categories', type=int, required=True, help='Number of categories to leave')

    args = parser.parse_args()
    prod_info = pd.read_csv(args.prod_info_csv)
    sample = top_categories_sample(prod_info, args.num_categories)
    sample.to_csv(args.output_file, index=False)
