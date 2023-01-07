import torch
from collections import Counter
from torchtext.vocab import Vocab
import re
import string
import numpy as np
import gensim
from nltk import ngrams

UNKNOWN_TOKEN = "<unk>"


class TweetsDataReader:
    def __init__(self, df, word_dict, lang, task=None):
        self.df = df
        '''
        if task == 'classify':
            # Might be changed to 'Label/ 'relevant'
            df['Label'] = 1
        '''
        self.word_dict = word_dict
        self.tweets = self.create_tweets()
        self.tweets_lengths = self.find_tweets_lengths()
        if lang == 'eng':
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings()
            self.tweets_word_idx = self.convert_tweets_to_dataset()
        if lang == 'heb':
            self.word_idx_mappings, self.word_vectors = self.init_word_embeddings_heb()
            self.tweets_word_idx = self.convert_tweets_to_dataset_heb()
        if lang == 'ar':
            self.word_idx_mappings, self.word_vectors = self.init_word_embeddings_ar()
            self.tweets_word_idx = self.convert_tweets_to_dataset_ar()
        self.labels = list(df['Label'])

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        return self.tweets_word_idx[index], self.tweets[index], self.tweets_lengths[index], self.labels[index]

    def create_tweets(self):
        tweets = []
        tweets_series = self.df['text']
        for tweet in tweets_series:
            tweets.append(self.clean_text(tweet))
        return tweets

    def clean_text(self, text):
        clean = str(text).lower()
        clean = clean.replace('\\', '')
        discluded = ['!', "'"]
        punc = [char for char in string.punctuation if char not in discluded]
        for char in punc:
            clean = clean.replace(char, '.')
        clean = re.sub('\.+', ' . ', clean)
        clean = re.sub(' +', ' ', clean)
        clean = clean.replace("'", '')
        clean = clean.replace("!", ' ! ')
        clean = clean.replace("  ", ' ')
        clean = clean.replace("\n", ' ')
        clean = clean.replace("\r", ' ')
        clean = re.sub("(\s!){3,}", " !!! ", clean)
        clean = re.sub("(\s!\s!)", " !! ", clean)
        clean = re.sub(' +', ' ', clean)
        clean = clean.strip()
        return clean.split(' ')

    def find_tweets_lengths(self):
        lengths_arr = []
        for tweet in self.tweets:
            lengths_arr.append(len(tweet))
        return lengths_arr

    def init_word_embeddings(self):
        global_glove = Vocab(Counter(self.word_dict), vectors="glove.twitter.27B.100d", specials=[UNKNOWN_TOKEN])
        return global_glove.stoi, global_glove.itos, global_glove.vectors

    def init_word_embeddings_heb(self):
        word_idx_mappings = dict()
        file1 = open('words_list.txt', 'r', encoding="utf8")
        word_vectors = np.load('words_vectors.npy')
        lines = file1.readlines()
        embeddings = torch.rand([1, 100], dtype=torch.double)  # For UNKNOWN word
        word_idx_mappings['*UNKNOWN*'] = 0
        dict_counter = 1
        for word_idx, word in enumerate(lines, start=1):
            word = word[:-1]
            if word in self.word_dict.keys():
                tensor_emb = torch.from_numpy(word_vectors[word_idx])
                tensor_emb = tensor_emb.reshape(1, 100)
                embeddings = torch.cat((embeddings, tensor_emb), dim=0)
                word_idx_mappings[word] = dict_counter
                dict_counter += 1
        return word_idx_mappings, embeddings

    def init_word_embeddings_ar(self):
        word_vectors = gensim.models.Word2Vec.load('full_grams_cbow_100_twitter.mdl')
        word_idx_mappings = {token: token_index for token_index, token in enumerate(word_vectors.wv.index2word)}
        word_idx_mappings['*UNKNOWN*'] = len(word_idx_mappings)
        embeddings = torch.from_numpy(word_vectors.wv.vectors)
        unknown_emb = torch.rand([1, 100], dtype=torch.float)  # For UNKNOWN word
        embeddings = torch.cat((embeddings, unknown_emb), dim=0)
        return word_idx_mappings, embeddings

    def convert_tweets_to_dataset(self):
        tweets_word_idx_list = list()
        for tweet in self.tweets:
            words_idx_list = []
            for word in tweet:
                if self.word_idx_mappings.get(word) is not None:
                    words_idx_list.append(self.word_idx_mappings.get(word))
                else:
                    words_idx_list.append(self.word_idx_mappings.get(UNKNOWN_TOKEN))
            tweets_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
        return tweets_word_idx_list

    def convert_tweets_to_dataset_heb(self):
        tweets_word_idx_list = list()
        for tweet in self.tweets:
            words_idx_list = []
            for word in tweet:
                if self.word_idx_mappings.get(word) is not None:
                    words_idx_list.append(self.word_idx_mappings.get(word))
                else:
                    words_idx_list.append(self.word_idx_mappings.get('*UNKNOWN*'))
            tweets_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
        return tweets_word_idx_list

    def convert_tweets_to_dataset_ar(self):
        tweets_word_idx_list = list()
        for tweet in self.tweets:
            words_idx_list = []
            for word in tweet:
                if self.word_idx_mappings.get(word) is not None:
                    words_idx_list.append(self.word_idx_mappings.get(word))
                else:
                    words_idx_list.append(self.word_idx_mappings.get('*UNKNOWN*'))
            tweets_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
        return tweets_word_idx_list
