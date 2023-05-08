import pandas as pd
import os

from gensim.models.word2vec import Word2Vec

from utils import get_blocks, get_sequence, parse_data


class Pipeline:
    def __init__(self, root, out, language, vocab_size=128, max_length=256):
        self.max_len = max_length
        self.root = root
        self.out = out
        self.language = language
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = os.path.join(self.root, "train.jsonl")
        self.dev_file_path = os.path.join(self.root, "valid.jsonl")
        self.test_file_path = os.path.join(self.root, "test.jsonl")
        self.size = vocab_size

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self):
        size = self.size

        trees = parse_data(self.train_file_path, self.language)

        embedding_path = os.path.join(self.out, 'train', 'embedding')
        os.makedirs(embedding_path, exist_ok=True)

        def trans_to_sequences(ast):
            sequence = []
            get_sequence(ast, sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(self.root + 'train/programs_ns.tsv')

        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3, max_final_vocab=10000)
        w2v.save(embedding_path + '/node_w2v_{}'.format(str(size)))

    # generate block sequences with index representations
    def generate_block_seqs(self, data_path, part):
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

        # if train, shuffle the data
        is_shuffle = part == 'train'

        trees = parse_data(data_path, self.language, is_shuffle=is_shuffle)
        trees['code'] = trees['code'].apply(trans2seq)
        os.makedirs(os.path.join(self.out, part), exist_ok=True)
        save_path = os.path.join(self.out, part)
        trees.to_pickle(save_path + '/blocks.pkl')

    def run(self):

        print('train word embedding...')
        self.dictionary_and_embedding()

        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


import argparse

parser = argparse.ArgumentParser(description="Set the params to preprocess the dataset")
parser.add_argument('--lang', default='c', choices=['c', 'java', 'python'], help='the language of the dataset')
parser.add_argument('--data_path', default='../dataset/', help='the path of the dataset')
parser.add_argument('--out_path', default='data/', help='the path of the output data')
args = parser.parse_args()

print(args)

os.makedirs(args.out_path, exist_ok=True)
ppl = Pipeline(args.data_path, args.out_path, args.lang)
ppl.run()
