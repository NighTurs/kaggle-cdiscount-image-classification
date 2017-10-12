import struct
import bson
import pandas as pd
from tqdm import tqdm
import argparse


def product_info(bson_path, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm() as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def save_product_info(product_info, file):
    product_info.to_csv(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', required=True, help='Path to bson with products')
    parser.add_argument('--without_categories', dest='with_categories', action='store_false',
                        help="Products don't have category_id field?")
    parser.set_defaults(with_categories=True)
    parser.add_argument('--output_file', required=True, help='File to save products info into')

    args = parser.parse_args()
    product_info = product_info(args.bson, args.with_categories)
    save_product_info(product_info, args.output_file)
