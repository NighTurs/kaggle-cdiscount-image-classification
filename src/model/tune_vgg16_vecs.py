import os
import argparse
import pandas as pd
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ..data.category_idx import map_categories
from ..data.train_split import train_slit
from .bcolz_iterator import BcolzIterator
from .vgg16_vecs import create_images_df

LOAD_MODEL = 'model.h5'
SNAPSHOT_MODEL = 'model_{epoch:02d}_{val_loss:.2f}.h5'
PREDICTIONS_FILE = 'predictions.csv'
MAX_PREDICTIONS_AT_TIME = 100000


def train_data(bcolz_root, prod_info, category_idx, only_first_image):
    images_df = create_images_df(prod_info, only_first_image)
    split = train_slit(prod_info.shape[0])

    prod_info['category_idx'] = map_categories(category_idx, prod_info['category_id'])
    prod_info['train'] = split

    cat_idxs = images_df.merge(prod_info, on='product_id', how='left')[['category_idx', 'train']]
    idxs = np.arange(cat_idxs.shape[0])
    train_idxs = idxs[cat_idxs['train']]
    valid_idxs = idxs[~cat_idxs['train']]

    num_classes = np.unique(cat_idxs['category_idx']).size

    train_it = BcolzIterator(bcolz_root=bcolz_root, x_idxs=train_idxs,
                             y=cat_idxs['category_idx'].iloc[train_idxs].as_matrix(),
                             num_classes=num_classes, seed=123)
    valid_it = BcolzIterator(bcolz_root=bcolz_root, x_idxs=valid_idxs,
                             y=cat_idxs['category_idx'].iloc[valid_idxs].as_matrix(),
                             num_classes=num_classes, seed=124)
    return train_it, valid_it, num_classes


def fit_model(train_it, valid_it, num_classes, models_dir, lr=0.001, batch_size=64, epochs=1):
    model_file = os.path.join(models_dir, LOAD_MODEL)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        inp = Input((512, 2, 2))
        x = Flatten()(inp)
        x = Dense(2000, activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inp, x)

    model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    np.random.seed(125)
    checkpointer = ModelCheckpoint(filepath=os.path.join(models_dir, SNAPSHOT_MODEL))
    model.fit_generator(train_it,
                        steps_per_epoch=train_it.samples / batch_size,
                        validation_data=valid_it,
                        validation_steps=valid_it.samples / batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer])


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
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true', dest='is_fit')
    parser.add_argument('--predict', action='store_true', dest='is_predict')
    parser.add_argument('--bcolz_root', required=True, help='VGG16 vecs bcolz root path')
    parser.add_argument('--prod_info_csv', required=True, help='Path to prod info csv')
    parser.add_argument('--category_idx_csv', required=True, help='Path to categories to index mapping csv')
    parser.add_argument('--models_dir', required=True, help='Output directory for models snapshots')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='Number of epochs')
    parser.add_argument('--only_first_image', dest='only_first_image', action='store_true',
                        help="Include only first image from each product")
    parser.set_defaults(only_first_image=False)

    args = parser.parse_args()
    if not os.path.isdir(args.models_dir):
        os.mkdir(args.models_dir)

    prod_info = pd.read_csv(args.prod_info_csv)
    category_idx = pd.read_csv(args.category_idx_csv)

    if args.is_fit:
        train_it, valid_it, num_classes = train_data(args.bcolz_root, prod_info, category_idx, args.only_first_image)
        fit_model(train_it, valid_it, num_classes, args.models_dir, args.lr, args.batch_size, args.epochs)
    if args.is_predict:
        out_df = predict(args.bcolz_root, prod_info, args.models_dir, args.only_first_image)
        out_df.to_csv(os.path.join(args.models_dir, PREDICTIONS_FILE), index=False)
