import numpy as np


class CombineIterator():
    def __init__(self, first_iterator, second_iterator):
        self.first_iterator = first_iterator
        self.second_iterator = second_iterator
        self.batch_size = first_iterator.batch_size + second_iterator.batch_size
        self.samples = first_iterator.samples + second_iterator.samples

    def next(self):
        first_out = self.first_iterator.next()
        second_out = self.second_iterator.next()
        if type(first_out[0]) is list:
            x = [np.concatenate((x1, x2)) for x1, x2 in zip(first_out[0], second_out[0])]
        else:
            x = np.concatenate((first_out[0], second_out[0]))
        y = np.concatenate((first_out[1], second_out[1]))
        return x, y

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
