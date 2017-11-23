import numpy as np
from keras.preprocessing.image import Iterator
from queue import Queue
from threading import Thread
import time


class MemmapIterator():
    def __init__(self, memmap_path, memmap_shape, images_df, num_classes=None, batch_size=32, shuffle=True, seed=None,
                 pool_wrokers=4, use_side_input=False):
        if seed:
            np.random.seed(seed)
        self.x = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=memmap_shape)
        self.images_df = images_df
        self.images_df_index = np.copy(self.images_df.index.values)
        self.images_df_num_imgs = np.copy(self.images_df.num_imgs.as_matrix())
        self.images_df_img_idx = np.copy(self.images_df.img_idx.as_matrix())
        self.has_y = 'category_idx' in images_df.columns
        if self.has_y:
            self.images_df_category_idx = np.copy(self.images_df.category_idx.as_matrix())
        del self.images_df
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_side_input = use_side_input
        self.samples = len(self.images_df_index)
        self.it = Iterator(self.samples, self.batch_size, self.shuffle, seed)
        self.queue = Queue(maxsize=40)
        self.stop_flag = False
        self.threads = []
        for i in range(pool_wrokers):
            thread = Thread(target=self.read_batches)
            thread.start()
            self.threads.append(thread)

    def read_batches(self):
        while True:
            if self.stop_flag == True:
                return
            with self.it.lock:
                index_array = next(self.it.index_generator)[0]
            m1 = np.zeros((len(index_array), *self.x.shape[1:]), dtype=np.float32)
            if self.use_side_input:
                m2 = np.zeros((len(index_array), 8), dtype=np.float32)

            if self.has_y:
                p = np.zeros(len(index_array), dtype=np.float32)

            for bi, i in enumerate(index_array):
                m1[bi] = self.x[self.images_df_index[i]]
                if self.use_side_input:
                    m2[bi, self.images_df_num_imgs[i] - 1] = 1
                    m2[bi, 4 + self.images_df_img_idx[i]] = 1
                if self.has_y:
                    # noinspection PyUnboundLocalVariable
                    p[bi] = self.images_df_category_idx[i]
            if self.use_side_input:
                inputs = [m1, m2]
            else:
                inputs = m1

            if self.has_y:
                self.queue.put((inputs, p))
            else:
                self.queue.put(inputs)

    def next(self):
        return self.queue.get()

    def terminate(self):
        self.stop_flag = True
        while True:
            try:
                while True:
                    self.queue.get(block=False)
            except:
                pass
            live_threads = 0
            for thread in self.threads:
                live_threads += 1 if thread.is_alive() else 0
            if live_threads == 0:
                return
            print('Threads running ', live_threads)
            for thread in self.threads:
                thread.join(timeout=5)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
