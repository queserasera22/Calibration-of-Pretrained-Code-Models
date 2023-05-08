from torch.utils.data import Dataset
import numpy as np

import const


class CNNDataset(Dataset):
    def __init__(self, src_sents, label, max_len, cuda=True):
        self._max_len = max_len
        self._src_sents = src_sents
        self._label = label
        self._len = len(label)
        self.cuda = cuda

    def classes(self):
        return self._label

    def get_batch_labels(self, idx):
        return np.array(self._label[idx])

    def get_batch_codes(self, idx):
        def pad_to_longest(insts, max_len):
            inst_data = insts + [const.PAD] * (max_len - len(insts))
            return inst_data
        batch_codes = pad_to_longest(self._src_sents[idx], self._max_len)
        return np.array(batch_codes)

    def __getitem__(self, idx):
        batch_codes = self.get_batch_codes(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_codes, batch_labels

    def __len__(self):
        return self._len
