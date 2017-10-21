import os
import argparse
import pandas as pd
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from ..data.category_idx import map_categories
from .bcolz_iterator import BcolzIterator
from .vgg16_vecs import create_images_df

LOAD_MODEL = 'model.h5'
SNAPSHOT_MODEL = 'model.h5'
LOG_FILE = 'training.log'
PREDICTIONS_FILE = 'predictions.csv'
MAX_PREDICTIONS_AT_TIME = 100000


def train_data(bcolz_root, bcolz_prod_info, sample_prod_info, train_split, category_idx, only_first_image, batch_size,
               shuffle=None, batch_seed=123):
    images_df = create_images_df(bcolz_prod_info, only_first_image)
    bcolz_prod_info['category_idx'] = map_categories(category_idx, bcolz_prod_info['category_id'])
    bcolz_prod_info = bcolz_prod_info.merge(train_split, on='product_id', how='left')
    cat_idxs = images_df.merge(bcolz_prod_info, on='product_id', how='left')[['product_id', 'category_idx', 'train']]
    if shuffle:
        np.random.seed(shuffle)
        perm = np.random.permutation(cat_idxs.shape[0])
        cat_idxs = cat_idxs.reindex(perm)
        cat_idxs.reset_index(drop=True, inplace=True)
    cat_idxs = cat_idxs[cat_idxs.product_id.isin(sample_prod_info.product_id)]
    idxs = cat_idxs.index.values
    train_idxs = idxs[cat_idxs['train']]
    valid_idxs = idxs[~cat_idxs['train']]
    num_classes = np.unique(cat_idxs['category_idx']).size

    train_it = BcolzIterator(bcolz_root=bcolz_root, x_idxs=train_idxs,
                             y=cat_idxs['category_idx'].loc[train_idxs].as_matrix(),
                             num_classes=num_classes, seed=batch_seed, batch_size=batch_size, shuffle=True)
    valid_it = BcolzIterator(bcolz_root=bcolz_root, x_idxs=valid_idxs,
                             y=cat_idxs['category_idx'].loc[valid_idxs].as_matrix(),
                             num_classes=num_classes, batch_size=batch_size, shuffle=False)
    return train_it, valid_it, num_classes


def fit_model(train_it, valid_it, num_classes, models_dir, lr=0.001, batch_size=64, epochs=1, mode=0):
    model_file = os.path.join(models_dir, LOAD_MODEL)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        if mode == 0:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 1:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(2048, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 2:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 3:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 4:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 5:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 6:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 7:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(2048, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 8:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 9:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(2048, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 10:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(4096, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)
        elif mode == 11:
            inp = Input((512, 2, 2))
            x = Flatten()(inp)
            x = Dense(4096, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dropout(0.5)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp, x)

    if mode == 6:
        model.compile(optimizer=SGD(lr=lr), loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
    else:
        model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

    np.random.seed(125)
    checkpointer = ModelCheckpoint(filepath=os.path.join(models_dir, SNAPSHOT_MODEL))
    csv_logger = CSVLogger(os.path.join(models_dir, LOG_FILE), append=True)
    model.fit_generator(train_it,
                        steps_per_epoch=train_it.samples / batch_size,
                        validation_data=valid_it,
                        validation_steps=valid_it.samples / batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, csv_logger])


def predict(bcolz_root, prod_info, models_dir, only_first_image, batch_size=200, top_k=20):
    model_file = os.path.join(models_dir, LOAD_MODEL)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        raise ValueError("Model doesn't exist")
    images_df = create_images_df(prod_info, only_first_image)
    it = BcolzIterator(bcolz_root=bcolz_root, x_idxs=np.arange(images_df.shape[0]), batch_size=batch_size,
                       shuffle=False)

    out_df = pd.DataFrame()
    steps = MAX_PREDICTIONS_AT_TIME // batch_size
    offset = 0

    while offset < images_df.shape[0]:
        preds = model.predict_generator(it, steps, verbose=1)
        top_k_preds = np.argpartition(preds, -top_k)[:, -top_k:]
        chunk = []
        for i in range(top_k_preds.shape[0]):
            product_id = images_df.iloc[offset + i]['product_id']
            img_idx = images_df.iloc[offset + i]['img_idx']
            for pred_idx in range(top_k):
                chunk.append((product_id, img_idx, top_k_preds[i, pred_idx], preds[i, top_k_preds[i, pred_idx]]))
        chunk_df = pd.DataFrame(chunk, columns=['product_id', 'img_idx', 'category_idx', 'prob'])
        out_df = pd.concat([out_df, chunk_df])
        offset += top_k_preds.shape[0]
        del top_k_preds
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true', dest='is_fit')
    parser.add_argument('--predict', action='store_true', dest='is_predict')
    parser.add_argument('--bcolz_root', required=True, help='VGG16 vecs bcolz root path')
    parser.add_argument('--bcolz_prod_info_csv', required=True,
                        help='Path to prod info csv with which VGG16 were generated')
    parser.add_argument('--sample_prod_info_csv', required=True, help='Path to sample prod info csv')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')
    parser.add_argument('--train_split_csv', required=True, help='Train split csv')
    parser.add_argument('--models_dir', required=True, help='Output directory for models snapshots')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='Number of epochs')
    parser.add_argument('--only_first_image', dest='only_first_image', action='store_true',
                        help="Include only first image from each product")
    parser.add_argument('--shuffle', type=int, default=None, required=False,
                        help='If products should be shuffled, provide seed')
    parser.set_defaults(only_first_image=False)
    parser.add_argument('--mode', type=int, default=0, required=False, help='Mode')
    parser.add_argument('--batch_seed', type=int, default=123, required=False, help='Batch seed')

    args = parser.parse_args()
    if not os.path.isdir(args.models_dir):
        os.mkdir(args.models_dir)

    bcolz_prod_info = pd.read_csv(args.bcolz_prod_info_csv)
    sample_prod_info = pd.read_csv(args.sample_prod_info_csv)
    train_split = pd.read_csv(args.train_split_csv)
    category_idx = pd.read_csv(args.category_idx_csv)

    if args.is_fit:
        train_it, valid_it, num_classes = train_data(args.bcolz_root, bcolz_prod_info, sample_prod_info,
                                                     train_split,
                                                     category_idx,
                                                     args.only_first_image,
                                                     args.batch_size, args.shuffle,
                                                     args.batch_seed)
        fit_model(train_it, valid_it, num_classes, args.models_dir, args.lr, args.batch_size, args.epochs, args.mode)
    if args.is_predict:
        out_df = predict(args.bcolz_root, bcolz_prod_info, args.models_dir, args.only_first_image)
        out_df.to_csv(os.path.join(args.models_dir, PREDICTIONS_FILE), index=False)
