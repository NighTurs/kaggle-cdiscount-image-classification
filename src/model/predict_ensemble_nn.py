import os
import argparse
import pandas as pd
from tqdm import tqdm
from keras.models import load_model

MODEL_FILE = 'model.h5'
TOP_PREDS = 10
PRODS_BATCH = 100000
CATEGORIES_SPLIT = 2000
PREDICTIONS_FILE = 'predictions.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csvs', nargs='+', required=True, help='Files with predictions of test dataset')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    parser.add_argument('--total_records', type=int, default=30950800,
                        help='Total number of records in prediction files')
    args = parser.parse_args()
    model = load_model(os.path.join(args.model_dir, MODEL_FILE))
    weights_left = model.get_layer('embedding_1').get_weights()
    weights_right = model.get_layer('embedding_2').get_weights()

    whole = []
    skiprows = 1
    with tqdm(total=args.total_records) as pbar:
        while skiprows < args.total_records + 1:
            all_preds = []
            for i, csv in enumerate(args.preds_csvs):
                weight_left = weights_left[0][i][0]
                weight_right = weights_right[0][i][0]
                preds = pd.read_csv(csv, skiprows=skiprows, nrows=TOP_PREDS * PRODS_BATCH,
                                    names=['product_id', 'img_idx', 'category_idx', 'prob'])
                preds.loc[preds.category_idx < CATEGORIES_SPLIT, 'prob'] = \
                    preds.loc[preds.category_idx < CATEGORIES_SPLIT, 'prob'] * weight_left
                preds.loc[preds.category_idx >= CATEGORIES_SPLIT, 'prob'] = \
                    preds.loc[preds.category_idx >= CATEGORIES_SPLIT, 'prob'] * weight_right
                all_preds.append(preds)
            all_preds = pd.concat(all_preds)
            all_preds = all_preds.groupby(['product_id', 'img_idx', 'category_idx'], as_index=False).sum()
            sum_preds = all_preds[['product_id', 'img_idx', 'prob']].groupby(['product_id', 'img_idx'],
                                                                             as_index=False).sum() \
                .rename(columns={'prob': 'prob_sum'})
            all_preds = all_preds.merge(sum_preds, on=['product_id', 'img_idx'], how='left')
            all_preds['prob'] = all_preds['prob'] / all_preds['prob_sum']
            all_preds = all_preds[['product_id', 'img_idx', 'category_idx', 'prob']]
            all_preds.sort_values('prob', inplace=True, ascending=False)
            all_preds = all_preds.groupby(['product_id', 'img_idx'], as_index=False).head(TOP_PREDS)

            print(all_preds.shape)
            whole.append(all_preds)

            skiprows += TOP_PREDS * PRODS_BATCH
            pbar.update(TOP_PREDS * PRODS_BATCH)

    pd.concat(whole).to_csv(os.path.join(args.model_dir, PREDICTIONS_FILE), index=False)
