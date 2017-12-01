import os
import argparse
import pandas as pd
import numpy as np
import keras.backend as K
from keras.preprocessing.image import Iterator
from src.data.category_idx import map_categories
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import merge
from keras.models import Model
from keras.initializers import Ones
from keras.optimizers import Adam
from keras.models import load_model
from keras.constraints import non_neg

N_CATEGORIES = 5270
CATEGORIES_SPLIT = 2000
MODEL_FILE = 'model.h5'


class SpecialIterator(Iterator):
    def __init__(self, images, categories, n_models, batch_size=32, shuffle=True, seed=None):
        self.x = images
        self.products = images[['product_id', 'img_idx']].drop_duplicates().sort_values(['product_id', 'img_idx'])
        self.categories = categories.sort_index()
        self.num_classes = N_CATEGORIES
        self.samples = self.products.shape[0]
        self.n_models = n_models
        super(SpecialIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        index_array = next(self.index_generator)[0]
        prods = self.products.iloc[index_array]
        pd = {(row.product_id, row.img_idx): i for i, row in enumerate(prods.itertuples())}
        cats = self.categories.loc[prods.product_id]
        images = prods.merge(self.x, on=['product_id', 'img_idx'], how='left')
        p = np.zeros((len(index_array), self.num_classes, self.n_models), dtype=np.float32)
        for row in images.itertuples():
            p[pd[(row.product_id, row.img_idx)], row.category_idx, row.model] = 0 if np.isnan(row.prob) else row.prob

        return [np.repeat(np.arange(self.n_models).reshape(1, self.n_models), len(index_array), axis=0),
                p[:, :CATEGORIES_SPLIT, :], p[:, CATEGORIES_SPLIT:, :]], cats['category_idx'].as_matrix()


def train_ensemble_nn(preds_csv_files, prod_info_csv, category_idx_csv, model_dir, lr, seed, batch_size, epochs):
    prod_info = pd.read_csv(prod_info_csv)
    category_idx = pd.read_csv(category_idx_csv)

    all_preds = []
    model_inx = {}
    for i, csv in enumerate(preds_csv_files):
        preds = pd.read_csv(csv)
        preds['model'] = i
        model_inx[i] = csv
        all_preds.append(preds)
    print('Assigned indexes to models: ', model_inx)
    all_preds = pd.concat(all_preds)

    n_models = len(preds_csv_files)

    categories = prod_info[prod_info.product_id.isin(all_preds.product_id.unique())][['product_id', 'category_id']]
    categories['category_idx'] = map_categories(category_idx, categories.category_id)
    categories = categories[['product_id', 'category_idx']]
    categories = categories.set_index('product_id')

    it = SpecialIterator(all_preds, categories, n_models, batch_size=batch_size, seed=seed, shuffle=True)

    model_file = os.path.join(model_dir, MODEL_FILE)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model_inp = Input(shape=(n_models,), dtype='int32')

        preds_cat1_inp = Input((CATEGORIES_SPLIT, n_models))
        preds_cat2_inp = Input((N_CATEGORIES - CATEGORIES_SPLIT, n_models))

        mul_cat1 = Embedding(n_models, 1, input_length=n_models, embeddings_initializer=Ones(),
                             embeddings_constraint=non_neg())(model_inp)
        mul_cat1 = Flatten()(mul_cat1)

        mul_cat2 = Embedding(n_models, 1, input_length=n_models, embeddings_initializer=Ones(),
                             embeddings_constraint=non_neg())(model_inp)
        mul_cat2 = Flatten()(mul_cat2)

        def op(x):
            z_left = x[0].dimshuffle(1, 0, 2) * x[1]
            z_right = x[2].dimshuffle(1, 0, 2) * x[3]
            z = K.concatenate([z_left, z_right], axis=0)
            v = K.sum(z, axis=-1)
            p = K.sum(v, axis=-2)
            return (v / p).dimshuffle(1, 0)

        x = merge([preds_cat1_inp, mul_cat1, preds_cat2_inp, mul_cat2], mode=op, output_shape=(N_CATEGORIES,))

        model = Model([model_inp, preds_cat1_inp, preds_cat2_inp], x)
    np.random.seed(seed)
    model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.fit_generator(it, steps_per_epoch=it.samples / it.batch_size, epochs=epochs)

    print('First {} categories model weights:'.format(CATEGORIES_SPLIT))
    print(model.get_layer('embedding_1').get_weights())
    print('Left categories model weights:'.format(CATEGORIES_SPLIT))
    print(model.get_layer('embedding_2').get_weights())

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model.save(os.path.join(model_dir, MODEL_FILE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csvs', nargs='+', required=True, help='Files with predictions of valid split')
    parser.add_argument('--prod_info_csv', required=True, help='Path to prod info csv')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')
    parser.add_argument('--seed', type=int, default=456, required=False, help='Learning seed')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=2000, required=False, help='Batch size')

    args = parser.parse_args()
    train_ensemble_nn(args.preds_csvs, args.prod_info_csv, args.category_idx_csv, args.model_dir, args.lr, args.seed,
                      args.batch_size, args.epochs)
