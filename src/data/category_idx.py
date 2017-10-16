import pandas as pd
import argparse


def create_category_idx(prod_info):
    category_stats = prod_info.groupby(by='category_id').size()
    category_stats.sort_values(ascending=False, inplace=True)
    category_idx = pd.DataFrame([(i, v) for i, v in enumerate(category_stats.index.values)],
                                columns=['category_idx', 'category_id'])
    return category_idx


def category_to_index_dict(category_idx):
    return {row.category_id: row.category_idx for row in category_idx.itertuples()}


def index_to_category_dict(category_idx):
    return {row.category_idx: row.category_id for row in category_idx.itertuples()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod_info_csv', required=True, help='Path to prod info csv')
    parser.add_argument('--output_file', required=True, help='File to save indexes into')
    args = parser.parse_args()
    prod_info = pd.read_csv(args.prod_info_csv)
    category_idx = create_category_idx(prod_info)
    category_idx.to_csv(args.output_file, index=False)
