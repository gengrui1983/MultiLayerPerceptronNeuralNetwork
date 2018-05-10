import h5py
import numpy as np


class DataLoader:
    def __init__(self, is_dev=False):
        training = self._open_h5_('/Users/chenjialu/Desktop/DL_Assignment1/Assignment-1-Dataset/train_128.h5', 'data')
        label = self._open_h5_('/Users/chenjialu/Desktop/DL_Assignment1/Assignment-1-Dataset/train_label.h5', 'label')

        n = np.random.np.random.randint(0, len(label), 60000)
        i = 40000

        self.training_dev = training[n[:i], :]
        self.label_dev = label[n[:i]]
        self.training_val = training[n[i:], :]
        self.label_val = label[n[i:]]

        if is_dev:
            n = np.random.randint(0, len(self.label), 1000)
            self.training = self.training[n]
            self.label = self.label[n]

    def load_data(self):
        return self

    @staticmethod
    def _open_h5_(filename, h_index):
        with h5py.File(filename, 'r') as H:
            data = np.copy(H[h_index])

            return data
