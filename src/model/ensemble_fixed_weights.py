import numpy as np
import pandas as pd
import os
import argparse
import itertools
from tqdm import tqdm
from collections import namedtuple

TOP_PREDS = 10
PREDICTIONS_FILE = 'predictions.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csvs', nargs='+', required=True, help='Files with predictions of test dataset')
    parser.add_argument('--weights', nargs='+', type=float, required=True, help='Weight of each model')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    args = parser.parse_args()

    if len(args.preds_csvs) != len(args.weights):
        raise ValueError('Count of weights should much count of csvs')

    preds_csvs = args.preds_csvs
    weights = args.weights
    model_dir = args.model_dir

    all_preds = []
    for i, csv in enumerate(preds_csvs):
        preds = pd.read_csv(csv, dtype={'product_id': np.int32,
                                        'img_idx': np.int8,
                                        'category_idx': np.int16,
                                        'prob': np.float32})
        preds['prob'] = preds['prob'] * weights[i]
        preds.sort_values(['product_id', 'img_idx'], inplace=True)
        all_preds.append(preds)

    prev_img = (0, 0)
    prev_cat = 0
    sum_prob = 0
    sum_all_probs = 0
    cat_prob = []

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Can't concatenate and sort all preds simultaneously because of memory problems
    def preds_gen(all_preds):
        iters = [pred.itertuples() for pred in all_preds]
        while True:
            rows = []
            for iter in iters:
                for i in range(TOP_PREDS):
                    rows.append(next(iter))
            rows.sort(key=lambda x: x.category_idx)
            for row in rows:
                yield row

    with tqdm(total=sum([preds.shape[0] for preds in all_preds])) as pbar, \
            open(os.path.join(model_dir, PREDICTIONS_FILE), 'w') as out:
        out.write('product_id,img_idx,category_idx,prob\n')
        for row in itertools.chain(preds_gen(all_preds),
                                   [namedtuple('Pandas', ['product_id', 'img_idx', 'category_idx', 'prob'])(0, 0, 0,
                                                                                                            0)]):
            product_id = row.product_id
            img_idx = row.img_idx
            category_idx = row.category_idx
            prob = row.prob
            if prev_img == (product_id, img_idx):
                if prev_cat == category_idx:
                    sum_prob += prob
                else:
                    cat_prob.append((prev_cat, sum_prob))
                    sum_all_probs += sum_prob
                    prev_cat = category_idx
                    sum_prob = prob
            else:
                cat_prob.append((prev_cat, sum_prob))
                sum_all_probs += sum_prob
                cat_prob = sorted(cat_prob, key=lambda x: x[1], reverse=True)[:TOP_PREDS]
                if prev_img != (0, 0):
                    for cat in cat_prob:
                        out.write('{},{},{},{}\n'.format(prev_img[0], prev_img[1], cat[0], cat[1] / sum_all_probs))
                prev_img = (product_id, img_idx)
                prev_cat = category_idx
                sum_prob = prob
                sum_all_probs = 0
                cat_prob = []
            pbar.update()