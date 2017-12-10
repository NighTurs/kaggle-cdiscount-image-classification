import numpy as np
import itertools
from collections import namedtuple
from keras.preprocessing.image import Iterator
from queue import Queue
from threading import Thread

class MultiMemmapIterator():
    def __init__(self, memmap_path, memmap_shape, images_df, num_classes=None, batch_size=32, shuffle=True, seed=None,
                 pool_wrokers=4, only_single=False, include_singles=True, max_images=2, use_side_input=True):
        if seed:
            np.random.seed(seed)
        self.x = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=memmap_shape)
        self.images_df = images_df.sort_values('product_id')
        self.images_df_index = np.copy(self.images_df.index.values)
        self.images_df_num_imgs = np.copy(self.images_df.num_imgs.as_matrix())
        self.images_df_img_idx = np.copy(self.images_df.img_idx.as_matrix())
        self.has_y = 'category_idx' in images_df.columns
        if self.has_y:
            self.images_df_category_idx = np.copy(self.images_df.category_idx.as_matrix())
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_images = max_images
        self.use_side_input = use_side_input

        self.smpls = []
        cur_index = []
        prev_product_id = -1
        for i, row in enumerate(
                itertools.chain(self.images_df.itertuples(), [namedtuple('Pandas', ['Index', 'product_id'])(0, 0)])):
            if prev_product_id != -1 and row.product_id != prev_product_id:
                if include_singles or len(cur_index) == 1:
                    self.smpls.extend([[idx] for idx in cur_index])
                if len(cur_index) > 1 and not only_single:
                    self.smpls.append(cur_index)
                cur_index = []
            prev_product_id = row.product_id
            cur_index.append(i)
        del self.images_df

        self.samples = len(self.smpls)
        self.rnd = np.random.RandomState(seed)
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
            m1 = np.zeros((len(index_array), self.max_images, *self.x.shape[1:]), dtype=np.float32)
            if self.use_side_input:
                m2 = np.zeros((len(index_array), self.max_images, 8), dtype=np.float32)

            if self.has_y:
                p = np.zeros(len(index_array), dtype=np.float32)

            bi = 0
            for smpl_idx in index_array:
                smpl = self.smpls[smpl_idx]

                for i in smpl:
                    cur_idx = 3 - self.images_df_img_idx[i]
                    m1[bi, cur_idx] = self.x[self.images_df_index[i]]
                    if self.use_side_input:
                        m2[bi, cur_idx, self.images_df_num_imgs[i] - 1] = 1
                        m2[bi, cur_idx, 4 + self.images_df_img_idx[i]] = 1

                if self.has_y:
                    # noinspection PyUnboundLocalVariable
                    p[bi] = self.images_df_category_idx[smpl[0]]
                bi += 1
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
