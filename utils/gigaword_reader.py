import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
import nltk


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs(vocab):
    data_train = {'context':[],'target':[]}
    data_dev = {'context':[],'target':[]}
    data_test = {'context':[],'target':[]}
    with open("data/gigaword_data/vocab.txt", encoding='utf-8') as f:
        for word in f:
            vocab.index_word(word.strip())
    with open("data/gigaword_data/train.article.txt", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_train['context'].append(line)
    with open("data/gigaword_data/train.title.txt", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_train['target'].append(line)
    assert len(data_train['context']) == len(data_train['target'])

    with open("data/gigaword_data/valid.article.filter.txt", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_dev['context'].append(line)
    with open("data/gigaword_data/valid.title.filter.txt", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_dev['target'].append(line)
    assert len(data_dev['context']) == len(data_dev['target'])

    with open("data/gigaword_data/input.txt", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_test['context'].append(line)
    with open("data/gigaword_data/task1_ref0.txt", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            data_test['target'].append(line)
    assert len(data_test['context']) == len(data_test['target'])

    return data_train, data_dev, data_test, vocab


def load_dataset():
    if(os.path.exists('data/gigaword/dataset_preproc.p')):
        print("LOADING gigaword")
        with open('data/gigaword/dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")

        data_tra, data_val, data_tst, vocab  = read_langs(vocab=Lang({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS", config.CLS1_idx:"CLS1", config.Y_idx:"Y"})) 
        with open('data/gigaword/dataset_preproc.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    for i in range(3):
        print("Examples:")
        print('[context]:', " ".join(data_tra['context'][i]))
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab


