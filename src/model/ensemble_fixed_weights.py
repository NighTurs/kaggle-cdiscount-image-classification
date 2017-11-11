import numpy as np
import pandas as pd
import os
import argparse

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
        all_preds.append(preds)

    all_preds = pd.concat(all_preds)
    print(all_preds.info(memory_usage='deep'))
    all_preds = all_preds.groupby(['product_id', 'img_idx', 'category_idx'], as_index=False).sum()
    all_preds.sort_values('prob', inplace=True, ascending=False)
    all_preds = all_preds.groupby(['product_id', 'img_idx'], as_index=False).head(TOP_PREDS)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    all_preds.to_csv(os.path.join(model_dir, PREDICTIONS_FILE), index=False)
