import bcolz
from keras.preprocessing.image import Iterator


class BcolzIterator(Iterator):
    def __init__(self, bcolz_root, x_idxs, y=None, num_classes=None, batch_size=32, shuffle=True, seed=None):
        self.x = bcolz.open(bcolz_root)
        self.x_idxs = x_idxs
        self.y = y
        self.num_classes = num_classes
        self.samples = len(self.x_idxs)
        super(BcolzIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
            if self.y is not None:
                return self.x[self.x_idxs[index_array[0]]], self.y[index_array[0]]
            else:
                return self.x[self.x_idxs[index_array[0]]]
