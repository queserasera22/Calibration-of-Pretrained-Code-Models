import json
import re

import torch

import argparse
import os

from collections import Counter

from const import *


def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


def split_doc(code):
    tokens = re.split('(\W+)', code)
    sub_tokens = []
    for token in tokens:
        sub_tokens.extend(split_on_uppercase(token, True))

    ret = ''
    for sub in sub_tokens:
        if sub is not ' ':
            ret = ret + sub + " "

    return ret.strip().split()


def split_on_uppercase(s, keep_contiguous=False):
    """example: s = 'split_onUppercaseNEW()' -->  ['split on', 'Uppercase', 'NEW()']  """
    s = s.replace("_", " ")
    string_length = len(s)
    is_lower_around = (lambda: s[i - 1].islower() or
                               string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i

    parts.append(s[start:])

    return parts


class Dictionary(object):
    def __init__(self, word2idx={}, idx_num=0):
        self.word2idx = word2idx
        self.idx = idx_num

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))


class Words(Dictionary):
    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, sents, vocab_size):
        words = [word for sent in sents for word in sent]
        counts = Counter(words)
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        size = min(len(sorted_words), vocab_size) if vocab_size is not None else len(sorted_words)
        for i in range(size):
            self._add(sorted_words[i][0])



class Labels(Dictionary):
    def __init__(self):
        super().__init__()

    def __call__(self, labels):
        _labels = set(labels)
        for label in _labels:
            self._add(label)


class Corpus(object):
    def __init__(self, path, save_data, max_len=256, vocab_size=10000):
        self.train = os.path.join(path, "train_sampled.txt")
        self.valid = os.path.join(path, "valid_sampled.txt")
        self.test = os.path.join(path, "test_sampled.txt")
        self._save_data = save_data

        self.w = Words()
        self.l = Labels()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.url_to_code = {}
        with open(os.path.join(path, "data.jsonl")) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                # self.url_to_code[js["idx"]] = js["func"]

                words = split_doc(js["func"])
                self.url_to_code[js["idx"]] = words
        # self.w(self.url_to_code.values(), self.vocab_size)

    def parse_data(self, _file, model):
        assert model in ['train', 'valid', 'test']
        _sents, _labels = [], []
        with open(_file, 'r', encoding='utf-8') as f:
            for line in f:
                url1, url2, label = line.split("\t")
                label = int(label)
                words = self.url_to_code[url1] + self.url_to_code[url2]

                if len(words) > self.max_len:
                    words = words[:self.max_len]

                _sents += [words]
                _labels += [label]

        if model == 'train':
            self.w(_sents, self.vocab_size)
            self.l(_labels)
            self.train_sents = _sents
            self.train_labels = _labels
        elif model == 'valid':
            self.valid_sents = _sents
            self.valid_labels = _labels
        else:
            self.test_sents = _sents
            self.test_labels = _labels

    def save(self):
        self.parse_data(self.train, 'train')
        self.parse_data(self.valid, 'valid')
        self.parse_data(self.test, 'test')

        data = {
            'max_len': self.max_len,
            'dict': {
                'train': self.w.word2idx,
                'vocab_size': len(self.w),
                # 'label': self.l.word2idx,
                'label_size': len(self.l),
            },
            'train': {
                'src': word2idx(self.train_sents, self.w.word2idx),
                # 'label': [self.l.word2idx[l] for l in self.train_labels]
                'label': self.train_labels
            },
            'valid': {
                'src': word2idx(self.valid_sents, self.w.word2idx),
                # 'label': [self.l.word2idx[l] for l in self.valid_labels]
                'label': self.valid_labels
            },
            'test': {
                'src': word2idx(self.test_sents, self.w.word2idx),
                # 'label': [self.l.word2idx[l] for l in self.test_labels]
                'label': self.test_labels
            }
        }

        torch.save(data, self._save_data)
        print('Finish dumping the data to file - [{}]'.format(self._save_data))
        print('words length - [{}]'.format(len(self.w)))
        print('label size - [{}]'.format(len(self.l)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--file-path', type=str, default="../dataset",
                        help='file path')
    parser.add_argument('--save-data', type=str, default="../dataset/corpus.pt",
                        help='path to save processed data')
    parser.add_argument('--max-lenth', type=int, default=256,
                        help='max length of sentence [default: 256]')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='max size of vocabulary [default: None]')
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data, args.max_lenth, args.vocab_size)
    corpus.save()
