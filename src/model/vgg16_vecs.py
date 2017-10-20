import pandas as pd
import numpy as np
import argparse
import os
import bcolz
from tqdm import tqdm
from .bson_iterator import BSONIterator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading


def compute_vgg16_vecs(bson_path, images_df, vecs_output_dir, save_step=100000):
    vgg_model = VGG16(include_top=False, input_shape=(3, 180, 180))

    if os.path.isdir(vecs_output_dir):
        vecs = bcolz.open(rootdir=vecs_output_dir)
        offset = vecs.shape[0]
    else:
        vecs = None
        offset = 0

    lock = threading.Lock()

    with open(bson_path, "rb") as train_bson_file, \
            tqdm(total=images_df.shape[0], initial=offset) as pbar:
        for i in range(offset, images_df.shape[0], save_step):
            gen = ImageDataGenerator(preprocessing_function=preprocess_input)
            batches = BSONIterator(bson_file=train_bson_file,
                                   images_df=images_df[i:(i + save_step)],
                                   num_class=0,  # doesn't matter here
                                   image_data_generator=gen,
                                   lock=lock,
                                   target_size=(180, 180),
                                   batch_size=220,
                                   shuffle=False,
                                   with_labels=False)
            x = AveragePooling2D()(vgg_model.output)
            model = Model(vgg_model.input, x)
            out_vecs = model.predict_generator(batches,
                                               steps=batches.samples / batches.batch_size,
                                               verbose=1)
            if not vecs:
                vecs = bcolz.carray(out_vecs, rootdir=vecs_output_dir, mode='w')
                vecs.flush()
            else:
                vecs.append(out_vecs)
                vecs.flush()
            pbar.update(save_step)


def create_images_df(product_info, only_first_image=False):
    rows = []
    for row in product_info.itertuples():
        for i in range(row.num_imgs):
            rows.append([row.product_id, i, row.offset, row.length])

    images_df = pd.DataFrame(rows, columns=['product_id', 'img_idx', 'offset', 'length'])
    if only_first_image:
        images_df = images_df[images_df.img_idx == 0]
        images_df = images_df.reset_index(drop=True)
    return images_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', required=True, help='Path to bson with products')
    parser.add_argument('--prod_info_csv', required=True, help='Path to prod info csv')
    parser.add_argument('--output_dir', required=True, help='Output directory for vectors')
    parser.add_argument('--save_step', type=int, required=True, help='Save computed vectors to disk each N steps')
    parser.add_argument('--only_first_image', dest='only_first_image', action='store_true',
                        help="Include only first image from each product")
    parser.add_argument('--shuffle', type=int, default=None, required=False,
                        help='If products should be shuffled, provide seed')
    parser.set_defaults(only_first_image=False)

    args = parser.parse_args()
    product_info = pd.read_csv(args.prod_info_csv)

    images_df = create_images_df(product_info, args.only_first_image)
    if args.shuffle:
        np.random.seed(args.shuffle)
        perm = np.random.permutation(images_df.shape[0])
        images_df = images_df.reindex(perm)

    compute_vgg16_vecs(args.bson, images_df, args.output_dir, args.save_step)
