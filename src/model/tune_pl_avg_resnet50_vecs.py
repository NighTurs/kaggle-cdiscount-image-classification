import os
import argparse
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Lambda
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from src.data.category_idx import map_categories
from src.model.multi_memmap_iterator import MultiMemmapIterator
from src.model.combine_iterator import CombineIterator
from src.model.resnet50_vecs import create_images_df

LOAD_MODEL = 'model.h5'
SNAPSHOT_MODEL = 'model.h5'
LOG_FILE = 'training.log'
PREDICTIONS_FILE = 'single_predictions.csv'
VALID_PREDICTIONS_FILE = 'valid_single_predictions.csv'
MAX_PREDICTIONS_AT_TIME = 50000


def create_train_df(prod_info, category_idx, shuffle=None):
    images_df = create_images_df(prod_info, False)
    prod_info['category_idx'] = map_categories(category_idx, prod_info['category_id'])
    images_df = images_df.merge(prod_info, on='product_id', how='left')[
        ['product_id', 'category_idx', 'img_idx', 'num_imgs']]
    if shuffle:
        np.random.seed(shuffle)
        perm = np.random.permutation(images_df.shape[0])
        images_df = images_df.reindex(perm)
        images_df.reset_index(drop=True, inplace=True)
    return images_df


def train_data(train_memmap_path,
               train_memmap_len,
               test_memmap_path,
               test_memmap_len,
               train_prod_info,
               train_pl_prod_info,
               test_pl_prod_info,
               train_split,
               category_idx,
               batch_size,
               shuffle=None,
               batch_seed=123,
               max_images=2,
               only_single=False,
               use_img_idx=False,
               include_singles=True):
    true_train_df = create_train_df(train_prod_info, category_idx, shuffle=shuffle)
    true_train_df = true_train_df.merge(train_split, on='product_id', how='left')
    num_classes = np.unique(true_train_df['category_idx']).size
    valid_df = true_train_df[~true_train_df['train']]
    del true_train_df

    pl_train_df = create_train_df(train_pl_prod_info, category_idx, shuffle=shuffle)
    pl_test_df = create_train_df(test_pl_prod_info, category_idx, shuffle=None)

    test_batch_size = int(batch_size * 0.25)
    train_batch_size = int(batch_size * 0.75)
    train_batch_size += 1 if test_batch_size + train_batch_size < batch_size else 0

    train_train_it = MultiMemmapIterator(memmap_path=train_memmap_path,
                                         memmap_shape=(train_memmap_len, 2048),
                                         images_df=pl_train_df,
                                         num_classes=num_classes,
                                         seed=batch_seed,
                                         batch_size=train_batch_size,
                                         only_single=only_single,
                                         include_singles=include_singles,
                                         max_images=max_images,
                                         pool_wrokers=4,
                                         shuffle=True,
                                         use_side_input=use_img_idx)

    train_test_it = MultiMemmapIterator(memmap_path=test_memmap_path,
                                        memmap_shape=(test_memmap_len, 2048),
                                        images_df=pl_test_df,
                                        num_classes=num_classes,
                                        seed=batch_seed,
                                        batch_size=test_batch_size,
                                        only_single=only_single,
                                        include_singles=include_singles,
                                        max_images=max_images,
                                        pool_wrokers=4,
                                        shuffle=True,
                                        use_side_input=use_img_idx)

    train_it = CombineIterator(train_train_it, train_test_it)

    valid_mul_it = MultiMemmapIterator(memmap_path=train_memmap_path,
                                       memmap_shape=(train_memmap_len, 2048),
                                       images_df=valid_df,
                                       num_classes=num_classes,
                                       seed=batch_seed,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       only_single=False,
                                       include_singles=False,
                                       max_images=4,
                                       pool_wrokers=1,
                                       use_side_input=use_img_idx)
    return train_it, valid_mul_it, num_classes


def fit_model(train_it, valid_mul_it, num_classes, models_dir, lr=0.001, batch_size=64, epochs=1, mode=0,
              seed=125):
    model_file = os.path.join(models_dir, LOAD_MODEL)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        if mode == 0:
            inp1 = Input((None, 2048))
            inp2 = Input((None, 8))
            x = concatenate([inp1, inp2])
            x = TimeDistributed(Dense(4096, activation='relu'))(x)
            x = BatchNormalization(axis=-1)(x)
            x = TimeDistributed(Dense(4096, activation='relu'))(x)
            x = BatchNormalization(axis=-1)(x)
            x = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(4096,))(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model([inp1, inp2], x)
        elif mode == 1:
            inp1 = Input((None, 2048))
            inp2 = Input((None, 8))
            x = concatenate([inp1, inp2])
            x = TimeDistributed(Dense(4096, activation='relu'))(x)
            x = BatchNormalization(axis=-1)(x)
            x = SimpleRNN(4096, activation='relu', recurrent_initializer='identity')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model([inp1, inp2], x)
        elif mode == 2:
            inp1 = Input((None, 2048))
            inp2 = Input((None, 8))
            x = concatenate([inp1, inp2])
            x = TimeDistributed(Dense(4096, activation='relu'))(x)
            x = BatchNormalization(axis=-1)(x)
            x = TimeDistributed(Dense(4096, activation='relu'))(x)
            x = BatchNormalization(axis=-1)(x)
            x = GRU(100, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model([inp1, inp2], x)
        elif mode == 3:
            inp1 = Input((None, 2048))
            inp1_com = Lambda(lambda x: K.max(x, axis=-2), output_shape=(2048,))(inp1)
            x = Dense(4096, activation='relu')(inp1_com)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(4096, activation='relu')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inp1, x)

    model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    np.random.seed(seed)
    checkpointer = ModelCheckpoint(filepath=os.path.join(models_dir, SNAPSHOT_MODEL))
    csv_logger = CSVLogger(os.path.join(models_dir, LOG_FILE), append=True)
    model.fit_generator(train_it,
                        steps_per_epoch=train_it.samples / train_it.batch_size,
                        validation_data=valid_mul_it,
                        validation_steps=valid_mul_it.samples / valid_mul_it.batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, csv_logger],
                        max_queue_size=10,
                        use_multiprocessing=False)

def predict(memmap_path, memmap_len, prod_info, sample_prod_info, models_dir, batch_size=200,
            shuffle=None, top_k=10, use_img_idx=True):
    model_file = os.path.join(models_dir, LOAD_MODEL)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        raise ValueError("Model doesn't exist")
    images_df = create_images_df(prod_info, False)
    images_df = images_df.merge(prod_info, on='product_id', how='left')[
        ['product_id', 'img_idx', 'num_imgs']]
    if shuffle:
        np.random.seed(shuffle)
        perm = np.random.permutation(images_df.shape[0])
        images_df = images_df.reindex(perm)
        images_df.reset_index(drop=True, inplace=True)
    if sample_prod_info is not None:
        images_df = images_df[images_df.product_id.isin(sample_prod_info.product_id)]
    images_df.sort_values('product_id', inplace=True)
    dfs = []
    offset = 0
    while offset < images_df.shape[0]:
        end_idx = min(images_df.shape[0], offset + MAX_PREDICTIONS_AT_TIME - 5)
        while end_idx < images_df.shape[0]:
            if images_df.iloc[end_idx - 1].product_id == images_df.iloc[end_idx].product_id:
                end_idx += 1
            else:
                break
        it = MultiMemmapIterator(memmap_path=memmap_path,
                                 memmap_shape=(memmap_len, 2048),
                                 images_df=images_df[offset:end_idx],
                                 batch_size=batch_size,
                                 pool_wrokers=1,
                                 only_single=False,
                                 include_singles=False,
                                 max_images=4,
                                 shuffle=False,
                                 use_side_input=use_img_idx)

        preds = model.predict_generator(it, it.samples / batch_size,
                                        verbose=1, max_queue_size=10)
        it.terminate()
        del it
        chunk = []
        for i, product_id in enumerate(images_df[offset:end_idx].product_id.unique()):
            top_k_preds = np.argpartition(preds[i], -top_k)[-top_k:]
            for pred_idx in range(top_k):
                chunk.append((product_id, 0, top_k_preds[pred_idx], preds[i, top_k_preds[pred_idx]]))

        chunk_df = pd.DataFrame(chunk, columns=['product_id', 'img_idx', 'category_idx', 'prob'])
        dfs.append(chunk_df)
        offset = end_idx
        del preds
        del chunk
    return pd.concat(dfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true', dest='is_fit')
    parser.add_argument('--predict', action='store_true', dest='is_predict')
    parser.add_argument('--predict_valid', action='store_true', dest='is_predict_valid')
    parser.add_argument('--memmap_path_train', required=True)
    parser.add_argument('--memmap_train_len', type=int, required=True)
    parser.add_argument('--memmap_path_test', required=True)
    parser.add_argument('--memmap_test_len', type=int, required=True)
    parser.add_argument('--train_prod_info_csv', required=True)
    parser.add_argument('--train_pl_prod_info_csv', required=True)
    parser.add_argument('--test_pl_prod_info_csv', required=True)
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
    parser.add_argument('--dont_use_img_idx', action='store_false', dest='use_img_idx')
    parser.set_defaults(use_img_idx=True)
    parser.set_defaults(two_outs=False)
    parser.add_argument('--max_images', type=int, default=2, required=False, help='Max images in train record')
    parser.add_argument('--only_single', action='store_true', dest='only_single')
    parser.set_defaults(only_single=False)
    parser.add_argument('--dont_include_singles', action='store_false', dest='include_singles')
    parser.set_defaults(include_singles=True)

    args = parser.parse_args()
    if not os.path.isdir(args.models_dir):
        os.mkdir(args.models_dir)

    train_prod_info = pd.read_csv(args.train_prod_info_csv)
    train_pl_prod_info = pd.read_csv(args.train_pl_prod_info_csv)
    test_pl_prod_info = pd.read_csv(args.test_pl_prod_info_csv)
    train_split = pd.read_csv(args.train_split_csv)
    category_idx = pd.read_csv(args.category_idx_csv)

    if args.is_fit:
        train_it, valid_mul_it, num_classes = train_data(args.memmap_path_train,
                                                         args.memmap_train_len,
                                                         args.memmap_path_test,
                                                         args.memmap_test_len,
                                                         train_prod_info,
                                                         train_pl_prod_info,
                                                         test_pl_prod_info,
                                                         train_split,
                                                         category_idx,
                                                         args.batch_size,
                                                         args.shuffle,
                                                         args.batch_seed,
                                                         args.max_images,
                                                         args.only_single,
                                                         args.use_img_idx,
                                                         args.include_singles)
        fit_model(train_it, valid_mul_it, num_classes, args.models_dir, args.lr, args.batch_size,
                  args.epochs,
                  args.mode,
                  args.batch_seed)
        train_it.first_iterator.terminate()
        train_it.second_iterator.terminate()
        valid_mul_it.terminate()
    elif args.is_predict:
        test_prod_info = test_pl_prod_info.drop('category_id', 1)
        out_df = predict(args.memmap_path_test, args.memmap_test_len, test_prod_info, test_prod_info, args.models_dir,
                         use_img_idx=args.use_img_idx)
        out_df.to_csv(os.path.join(args.models_dir, PREDICTIONS_FILE), index=False)
    elif args.is_predict_valid:
        only_valids = train_prod_info[
            train_prod_info.product_id.isin(train_split[train_split.train == False].product_id)]
        out_df = predict(args.memmap_path_train, args.memmap_train_len, train_prod_info, only_valids, args.models_dir,
                         shuffle=args.shuffle, use_img_idx=args.use_img_idx)
        out_df.to_csv(os.path.join(args.models_dir, VALID_PREDICTIONS_FILE), index=False)
