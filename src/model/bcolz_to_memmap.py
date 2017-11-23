import argparse
import bcolz
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bcolz_path', required=True, help='Path to bcolz with vectors')
    parser.add_argument('--memmap_path', required=True, help="Write memmap to path")

    args = parser.parse_args()

    bcolz_path = args.bcolz_path
    memmap_path = args.memmap_path

    a = bcolz.open(bcolz_path)
    b = np.memmap(memmap_path, dtype='float32', mode='w+', shape=a.shape)

    with tqdm(total=a.shape[0]) as pbar:
        batch_size = 100000
        for i in range(0, a.shape[0], batch_size):
            chunk = a[i:(i + batch_size)]
            b[i:(i + batch_size)] = chunk
            b.flush()
            pbar.update(chunk.shape[0])