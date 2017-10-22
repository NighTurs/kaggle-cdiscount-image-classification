import pandas as pd
import argparse
import numpy as np
from ..data.category_idx import index_to_category_dict


def max_prob_category(df):
    i = df.prob.argmax()
    return df.category_idx.loc[i]


def form_submission(preds, category_idx):
    preds.sort_values('prob', ascending=False, inplace=True)
    taken_prods = set()
    d = index_to_category_dict(category_idx)
    products = []
    category_id = []
    for row in preds.itertuples():
        if row.product_id in taken_prods:
            continue
        taken_prods.add(row.product_id)
        products.append(row.product_id)
        category_id.append(d[row.category_idx])
    submission = pd.Series(category_id, index=products)
    submission.rename('category_id', inplace=True)
    submission.index.rename(name='_id', inplace=True)
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csv', required=True, help='File with predictions')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')
    parser.add_argument('--output_file', required=True, help='File to save submission into')

    args = parser.parse_args()
    preds = pd.read_csv(args.preds_csv, dtype={'category_idx': np.int16, 'prob': np.float32, 'product_id': np.int32,
                                               'img_idx': np.int8})
    category_idx = pd.read_csv(args.category_idx_csv)
    submission = form_submission(preds, category_idx)
    submission.to_csv(args.output_file, header=True)
