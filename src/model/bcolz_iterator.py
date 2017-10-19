import numpy as np
import bcolz
from keras.preprocessing.image import Iterator

CHUNK_SIZE = 100000


class BcolzIterator():
    def __init__(self, bcolz_root, x_idxs, y=None, num_classes=None, batch_size=32, shuffle=True, seed=None):
        self.x = bcolz.open(bcolz_root)
        self.x_idxs = x_idxs
        self.y = y
        self.num_classes = num_classes
        self.samples = len(self.x_idxs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if seed:
            np.random.seed(seed)
        self.chunk_idx = -1
        self.get_chunk()

    def get_chunk(self):
        self.chunk_idx += 1
        if CHUNK_SIZE * self.chunk_idx >= len(self.x_idxs):
            self.chunk_idx = 0
        idxs = self.x_idxs[(CHUNK_SIZE * self.chunk_idx):(CHUNK_SIZE * self.chunk_idx + CHUNK_SIZE)]
        self.chunk_x = self.x[idxs]
        self.chunk_y = self.y[(CHUNK_SIZE * self.chunk_idx):(CHUNK_SIZE * self.chunk_idx + CHUNK_SIZE)]
        self.chunk_seen = 0
        self.it = Iterator(len(idxs), self.batch_size, self.shuffle, None)

    def next(self):
        if self.chunk_x.shape[0] <= self.chunk_seen:
            self.get_chunk()
        index_array = next(self.it.index_generator)
        if self.y is not None:
            out = self.chunk_x[index_array[0]], self.chunk_y[index_array[0]]
        else:
            out = self.chunk_x[index_array[0]]
        self.chunk_seen += len(index_array[0])
        return out

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
