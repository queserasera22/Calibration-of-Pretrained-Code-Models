import json

import pandas as pd
import os
import sys
import warnings
from utils import get_blocks, get_sequence, parse_data
from gensim.models.word2vec import Word2Vec

warnings.filterwarnings('ignore')


# create clone pairs
def read_pairs(file_path):
    pairs = pd.read_csv(file_path, names=['id1', 'id2', 'label'], header=None, sep="\t")
    return pairs


class Pipeline:
    def __init__(self, root, out, language, max_length=512):
        self.root = root
        self.out = out
        self.language = language
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = self.root + '/train_sampled.txt'
        self.dev_file_path = self.root + '/valid_sampled.txt'
        self.test_file_path = self.root + '/test_sampled.txt'
        self.size = 128
        self.max_len = max_length

    # parse source code
    def parse_source(self):
        os.makedirs(self.out, exist_ok=True)
        source = parse_data(os.path.join(self.root, "data.jsonl"), self.language, self.max_len)
        self.sources = source
        return source

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size):
        self.size = size
        data_path = self.out + self.language + '/'
        pairs = read_pairs(self.train_file_path)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('idx', drop=False).loc[train_ids]
        os.makedirs(data_path + 'train/embedding', exist_ok=True)

        def trans_to_sequences(ast):
            sequence = []
            get_sequence(ast, sequence)
            return sequence

        corpus = trees['func'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['func'] = pd.Series(str_corpus)

        embedding_path = os.path.join(self.out, 'train', 'embedding')
        os.makedirs(embedding_path, exist_ok=True)
        w2v = Word2Vec(corpus,  size=size, workers=16, sg=1, max_final_vocab=10000)
        w2v.save(embedding_path + '/node_w2v_{}'.format(str(size)))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        embedding_path = os.path.join(self.out, 'train', 'embedding')
        word2vec = Word2Vec.load(embedding_path + '/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            get_blocks(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = pd.DataFrame(self.sources, copy=True)
        trees['func'] = trees['func'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self, data_path, part):
        pairs = read_pairs(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        df.dropna(inplace=True)

        path = os.path.join(self.out, part)
        os.makedirs(path, exist_ok=True)
        df.to_pickle(path + '/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source()

        # print('train word embedding...')
        # self.dictionary_and_embedding(128)

        print('generate block sequences...')
        self.generate_block_seqs()

        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')


import argparse
parser = argparse.ArgumentParser(description="Set the params to preprocess the dataset")
parser.add_argument('--lang', default='c', choices=['c', 'java', 'python'], help='the language of the dataset')
parser.add_argument('--data_path', default='../dataset/', help='the path of the dataset')
parser.add_argument('--out_path', default='data/', help='the path of the output data')
parser.add_argument('--max_len', type=int, default=256, help='the max length of the block sequence')
args = parser.parse_args()

print(args)

os.makedirs(args.out_path, exist_ok=True)
ppl = Pipeline(args.data_path, args.out_path, args.lang, args.max_len)
ppl.run()
