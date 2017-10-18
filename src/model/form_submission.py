import pandas as pd
import argparse
from ..data.category_idx import index_to_category_dict


def max_prob_category(df):
    i = df.prob.argmax()
    return df.category_idx.loc[i]


def form_submission(preds, category_idx):
    submission = preds.groupby('product_id')[['category_idx', 'prob']].agg(max_prob_category)
    d = index_to_category_dict(category_idx)
    submission['category_id'] = [d[x] for x in submission['category_idx']]
    submission.index.rename(name='_id', inplace=True)
    return submission['category_id']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csv', required=True, help='File with predictions')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')
    parser.add_argument('--output_file', required=True, help='File to save submission into')

    args = parser.parse_args()
    preds = pd.read_csv(args.preds_csv)
    category_idx = pd.read_csv(args.category_idx_csv)
    submission = form_submission(preds, category_idx)
    submission.to_csv(args.output_file, header=True)
