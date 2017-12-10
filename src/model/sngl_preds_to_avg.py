import pandas as pd
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from collections import namedtuple

TOP_K = 10

def sngl_preds_to_avg(preds):
    preds.sort_values(['product_id', 'category_idx'], inplace=True)
    cur = (0, 0)
    acc = 0
    prod_cats = []
    chunks = []

    with tqdm(total=preds.shape[0]) as pbar:
        for row in itertools.chain(preds.itertuples(),
                                   [namedtuple('Pandas', ['product_id', 'img_idx', 'category_idx', 'prob'])(0, 0, 0,
                                                                                                            0)]):
            if cur == (row.product_id, row.category_idx):
                acc += row.prob
            else:
                prod_cats.append((cur[1], acc))
                if row.product_id != cur[0]:
                    if cur != (0, 0):
                        prod_cats.sort(key=lambda x: x[1], reverse=True)
                        s = sum([x[1] for x in prod_cats])
                        for t in prod_cats[:TOP_K]:
                            chunks.append((cur[0], 0, t[0], t[1] / s))
                    prod_cats = []
                cur = (row.product_id, row.category_idx)
                acc = row.prob
            pbar.update(1)

    return pd.DataFrame(chunks, columns=['product_id', 'img_idx', 'category_idx', 'prob'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csv', required=True, help='File with predictions')
    parser.add_argument('--output_file', required=True, help='File to save submission into')

    args = parser.parse_args()
    preds = pd.read_csv(args.preds_csv, dtype={'category_idx': np.int16, 'prob': np.float32, 'product_id': np.int32,
                                               'img_idx': np.int8})
    out = sngl_preds_to_avg(preds)
    out.to_csv(args.output_file, header=True)