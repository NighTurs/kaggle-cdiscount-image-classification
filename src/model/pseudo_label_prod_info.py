import pandas as pd
import argparse
from src.data.category_idx import index_to_category_dict


def pick_top_category(preds, category_map):
    ordered_preds = preds.sort_values('prob', ascending=False)
    ordered_preds.fillna(0, inplace=True)
    taken_products = set()
    tuples = []
    for row in ordered_preds.itertuples():
        if row.product_id in taken_products:
            continue
        taken_products.add(row.product_id)
        tuples.append((row.product_id, category_map[row.category_idx]))
    return pd.DataFrame(tuples, columns=['product_id', 'category_id'], dtype='int64').sort_values('product_id')


def create_pl_prod_infos(train_prod_info_csv, test_prod_info_csv, valid_preds_csv, test_preds_csv, pl_train_prod_info,
                         pl_test_prod_info, category_idx_csv):
    train_prod_info = pd.read_csv(train_prod_info_csv)
    test_prod_info = pd.read_csv(test_prod_info_csv)
    valid_preds = pd.read_csv(valid_preds_csv)
    test_preds = pd.read_csv(test_preds_csv)
    category_idx = pd.read_csv(category_idx_csv)

    category_map = index_to_category_dict(category_idx)

    test_preds = pick_top_category(test_preds, category_map)
    test_prod_info = test_prod_info.merge(test_preds, on='product_id', how='left')
    test_prod_info.to_csv(pl_test_prod_info, index=False)

    valid_preds = pick_top_category(valid_preds, category_map)
    train_prod_info.loc[train_prod_info.product_id.isin(valid_preds.product_id), 'category_id'] = valid_preds[
        'category_id'].as_matrix()
    train_prod_info.to_csv(pl_train_prod_info, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prod_info', required=True, help='Train product info')
    parser.add_argument('--test_prod_info', required=True, help='Test product info')
    parser.add_argument('--valid_preds', required=True, help='Valid split predictions')
    parser.add_argument('--test_preds', required=True, help='Test predictions')
    parser.add_argument('--pl_train_prod_info', required=True, help='Pseudo labeling train product info output file')
    parser.add_argument('--pl_test_prod_info', required=True, help='Pseudo labeling test product info output file')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')

    args = parser.parse_args()
    create_pl_prod_infos(args.train_prod_info, args.test_prod_info, args.valid_preds, args.test_preds,
                         args.pl_train_prod_info, args.pl_test_prod_info, args.category_idx_csv)
