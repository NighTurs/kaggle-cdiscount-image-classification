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
from keras.layers import Flatten
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from src.data.category_idx import map_categories
from src.model.multi_memmap_iterator import MultiMemmapIterator
from src.model.resnet50_vecs import create_images_df

LOAD_MODEL = 'model.h5'
SNAPSHOT_MODEL = 'model.h5'
LOG_FILE = 'training.log'
PREDICTIONS_FILE = 'single_predictions.csv'
VALID_PREDICTIONS_FILE = 'valid_single_predictions.csv'
MAX_PREDICTIONS_AT_TIME = 50000


def train_data(memmap_path, memmap_len, bcolz_prod_info, sample_prod_info, train_split, category_idx, batch_size,
               shuffle=None, batch_seed=123, max_images=2, only_single=False, use_img_idx=False):
    images_df = create_images_df(bcolz_prod_info, False)
    bcolz_prod_info['category_idx'] = map_categories(category_idx, bcolz_prod_info['category_id'])
    bcolz_prod_info = bcolz_prod_info.merge(train_split, on='product_id', how='left')
    images_df = images_df.merge(bcolz_prod_info, on='product_id', how='left')[
        ['product_id', 'category_idx', 'img_idx', 'num_imgs', 'train']]
    if shuffle:
        np.random.seed(shuffle)
        perm = np.random.permutation(images_df.shape[0])
        images_df = images_df.reindex(perm)
        images_df.reset_index(drop=True, inplace=True)
    images_df = images_df[images_df.product_id.isin(sample_prod_info.product_id)]
    train_df = images_df[images_df['train']]
    valid_df = images_df[~images_df['train']]
    num_classes = np.unique(images_df['category_idx']).size

    train_it = MultiMemmapIterator(memmap_path=memmap_path,
                                   memmap_shape=(memmap_len, 512, 2, 2),
                                   images_df=train_df,
                                   num_classes=num_classes,
                                   seed=batch_seed,
                                   batch_size=batch_size,
                                   only_single=only_single,
                                   include_singles=False,
                                   max_images=max_images,
                                   pool_wrokers=4,
                                   shuffle=True,
                                   use_side_input=use_img_idx)
    valid_mul_it = MultiMemmapIterator(memmap_path=memmap_path,
                                       memmap_shape=(memmap_len, 512, 2, 2),
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
    valid_sngl_it = MultiMemmapIterator(memmap_path=memmap_path,
                                        memmap_shape=(memmap_len, 512, 2, 2),
                                        images_df=valid_df,
                                        num_classes=num_classes,
                                        seed=batch_seed,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        only_single=True,
                                        include_singles=True,
                                        max_images=1,
                                        pool_wrokers=1,
                                        use_side_input=use_img_idx)
    return train_it, valid_mul_it, valid_sngl_it, num_classes


def fit_model(train_it, valid_mul_it, valid_sngl_it, num_classes, models_dir, lr=0.001, batch_size=64, epochs=1, mode=0,
              seed=125):
    model_file = os.path.join(models_dir, LOAD_MODEL)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        if mode == 0:
            inp1 = Input((None, 512, 2, 2))
            inp1_flat = TimeDistributed(Flatten())(inp1)
            inp1_flat = Lambda(lambda x: K.max(x, axis=-2), output_shape=(2048,))(inp1_flat)
            x = Dense(4096, activation='relu')(inp1_flat)
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
                        steps_per_epoch=train_it.samples / batch_size,
                        validation_data=valid_mul_it,
                        validation_steps=valid_mul_it.samples / batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, csv_logger],
                        max_queue_size=10,
                        use_multiprocessing=False)

    with open(os.path.join(models_dir, LOG_FILE), "a") as file:
        file.write('Multi {}\n'.format(model.evaluate_generator(valid_mul_it, steps=valid_mul_it.samples / batch_size)))
        file.write(
            'Single {}\n'.format(model.evaluate_generator(valid_sngl_it, steps=valid_sngl_it.samples / batch_size)))


def predict(memmap_path, memmap_len, prod_info, sample_prod_info, models_dir, batch_size=200,
            shuffle=None, top_k=10, use_img_idx=False):
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
                                 memmap_shape=(memmap_len, 512, 2, 2),
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
    parser.add_argument('--memmap_path', required=True, help='VGG16 vecs memmap path')
    parser.add_argument('--prod_info_csv', required=True,
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
    parser.add_argument('--use_img_idx', action='store_true', dest='use_img_idx')
    parser.set_defaults(use_img_idx=False)
    parser.add_argument('--memmap_len', type=int, required=True, help='Number of rows in memmap')
    parser.set_defaults(two_outs=False)
    parser.add_argument('--max_images', type=int, default=2, required=False, help='Max images in train record')
    parser.add_argument('--only_single', action='store_true', dest='only_single')
    parser.set_defaults(only_single=False)

    args = parser.parse_args()
    if not os.path.isdir(args.models_dir):
        os.mkdir(args.models_dir)

    prod_info = pd.read_csv(args.prod_info_csv)
    sample_prod_info = pd.read_csv(args.sample_prod_info_csv)
    train_split = pd.read_csv(args.train_split_csv)
    category_idx = pd.read_csv(args.category_idx_csv)

    if args.is_fit:
        train_it, valid_mul_it, valid_sngl_it, num_classes = train_data(args.memmap_path,
                                                                        args.memmap_len,
                                                                        prod_info,
                                                                        sample_prod_info,
                                                                        train_split,
                                                                        category_idx,
                                                                        args.batch_size,
                                                                        args.shuffle,
                                                                        args.batch_seed,
                                                                        args.max_images,
                                                                        args.only_single,
                                                                        args.use_img_idx)
        fit_model(train_it, valid_mul_it, valid_sngl_it, num_classes, args.models_dir, args.lr, args.batch_size,
                  args.epochs,
                  args.mode,
                  args.batch_seed)
        train_it.terminate()
        valid_mul_it.terminate()
        valid_sngl_it.terminate()
    elif args.is_predict:
        out_df = predict(args.memmap_path, args.memmap_len, prod_info, sample_prod_info, args.models_dir,
                         use_img_idx=args.use_img_idx)
        out_df.to_csv(os.path.join(args.models_dir, PREDICTIONS_FILE), index=False)
    elif args.is_predict_valid:
        only_valids = prod_info[
            prod_info.product_id.isin(train_split[train_split.train == False].product_id)]
        out_df = predict(args.memmap_path, args.memmap_len, prod_info, only_valids, args.models_dir,
                         shuffle=args.shuffle, use_img_idx=args.use_img_idx)
        out_df.to_csv(os.path.join(args.models_dir, VALID_PREDICTIONS_FILE), index=False)
