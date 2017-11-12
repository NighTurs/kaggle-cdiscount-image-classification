import pandas as pd
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from collections import namedtuple
from ..data.category_idx import index_to_category_dict

DEFAULT_PROB = 0.001

def max_prob_category(df):
    i = df.prob.argmax()
    return df.category_idx.loc[i]


def form_submission(preds, category_idx):
    preds.sort_values(['product_id', 'category_idx'], ascending=False, inplace=True)
    d = index_to_category_dict(category_idx)
    cur = (0, 0)
    acc = 1
    imgs = 0
    max_acc = 0
    max_imgs = 1
    max_cat = 0
    products = []
    category_id = []
    with tqdm(total=preds.shape[0]) as pbar:
        for row in itertools.chain(preds.itertuples(),
                                   [namedtuple('Pandas', ['product_id', 'img_idx', 'category_idx', 'prob'])(0, 0, 0, 0)]):
            if cur == (row.product_id, row.category_idx):
                acc *= row.prob
                imgs += 1
            else:
                if row.product_id == cur[0]:
                    while imgs > max_imgs:
                        max_imgs += 1
                        max_acc *= DEFAULT_PROB
                    while max_imgs > imgs:
                        imgs += 1
                        acc *= DEFAULT_PROB
                    max_imgs = imgs
                    if max_acc < acc:
                        max_acc = acc
                        max_cat = cur[1]
                else:
                    if cur != (0, 0):
                        products.append(cur[0])
                        category_id.append(d[max_cat])
                    max_acc = 0
                    max_imgs = 1
                cur = (row.product_id, row.category_idx)
                acc = row.prob
                imgs = 1
            pbar.update(1)
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
