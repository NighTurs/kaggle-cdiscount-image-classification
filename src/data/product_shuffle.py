import argparse
import pandas as pd
import itertools
import bcolz
import numpy as np
from tqdm import tqdm
import src.model.resnet50_vecs as vecs


def create_images_df(prod_info, train_split, seed):
    images_df = vecs.create_images_df(prod_info, False)[['product_id', 'img_idx']]

    np.random.seed(seed)
    perm = np.random.permutation(images_df.shape[0])
    images_df = images_df.reindex(perm)
    images_df.reset_index(drop=True, inplace=True)

    train_split.sort_values('product_id', inplace=True)
    products_train = train_split.product_id[train_split.train == True]
    products_test = train_split.product_id[train_split.train == False]
    np.random.seed(seed)
    p_map = {k: v for k, v in itertools.chain(zip(products_train, np.random.permutation(len(products_train))),
                                              zip(products_test, len(products_train) +
                                                  np.random.permutation(len(products_test))))}
    images_df['pos'] = images_df.product_id.apply(lambda x: p_map[x])
    return images_df.sort_values(['pos', 'img_idx'], inplace=True)[['product_id', 'img_idx']]


def reorder_bcolz(permutation, bcolz_input, bcolz_output):
    c = bcolz.open(bcolz_input)
    perm = np.copy(permutation)
    CHUNK = 200000
    b = None
    with tqdm(total=c.shape[0]) as pb:
        for step in range(0, c.shape[0], CHUNK):
            idxs = perm[step:(step + CHUNK)]
            tp = sorted(enumerate(idxs), key=lambda x: x[1])
            chunk = c[[x[1] for x in tp]]
            chunk = chunk[[y[0] for y in sorted(enumerate([x[0] for x in tp]), key=lambda x: x[1])]]
            if type(b) == type(None):
                b = bcolz.carray(chunk,
                                 rootdir=bcolz_output)
                b.flush()
            else:
                b.append(chunk)
                b.flush()
            pb.update(len(idxs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_product_info', required=True, help='Path train product info')
    parser.add_argument('--train_split', required=True, help="Train split csv")
    parser.add_argument('--bcolz_input', required=True, help='Path to bcolz with vectors')
    parser.add_argument('--bcolz_output', required=True, help='Path to output reordered bcolz vectors')
    parser.add_argument('--shuffle', type=int, required=True, help='Seed to shuffle products')

    args = parser.parse_args()

    prod_info = pd.read_csv(args.train_product_info)
    train_split = pd.read_csv(args.train_split)

    images_df = create_images_df(prod_info, train_split, args.shuffle)
    permutation = images_df.index.values
    del images_df
    reorder_bcolz(permutation, args.bcolz_input, args.bcolz_output)
