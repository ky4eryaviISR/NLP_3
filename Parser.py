import copy
from abc import abstractmethod
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

WORDS = ["car", "bus", "hospital", "hotel", "gun", "bomb", "fox", "table", "bowl", "guitar", "piano"]

CONTENT_CLASSES = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
                   'VBP', 'VBZ', 'WRB'}
LEMMA_MIN = 100


class Parser(object):

    def __init__(self, path):
        self.word_att = {}
        self.context_count = {}
        self.word_set = self.word_index = self.index_word = None
        print(f"{datetime.now()}:Build Dictionary")
        self.get_word_index_dict(path)
        print(f"{datetime.now()}:File Parsing")
        self.sentences = self.load_txt(path)

    def PMI_smooth(self, word, context, total):
        P_xy = self.word_att[word][context] / total
        P_x = sum(self.word_att[word].values()) / total
        P_y = self.context_count[context] / total
        return np.log(P_xy / (P_x * P_y))

    @abstractmethod
    def get_context(self, s, sen_index=None):
        pass

    def get_word_index_dict(self,vocabulary):
        lemma_count = {}
        with open(vocabulary, encoding='utf8') as f:
            for line in f:
                # if we get the end of the sentence add it to all sentences
                if line == '\n':
                    continue
                lemma = line.split('\t')[2]
                if line.split('\t')[4] in CONTENT_CLASSES:
                    lemma_count[lemma] = 1 if lemma not in lemma_count \
                        else lemma_count[lemma] + 1
        self.lemma_cnt = lemma_count
        self.word_set = [lemma for lemma, count in self.lemma_cnt.items() if count >= LEMMA_MIN]
        # index for each lemma
        self.word_index = {k: index for index, k in enumerate(self.word_set)}
        self.index_word = {v: k for k, v in self.word_index.items()}

    def calculate_pmi(self):

        total_word_att = 0
        for word in self.word_att.keys():
            total_word_att += sum(self.word_att[word].values())
        word_att_pmi = copy.deepcopy(self.word_att)
        for word in self.word_att.keys():
            att_vec = word_att_pmi[word] = {k: v for k, v in sorted(self.word_att[word].items(),
                                                                    key=lambda item: item[1],
                                                                    reverse=True)[0:100]}
            for att, count in list(att_vec.items()):
                pmi = self.PMI_smooth(word, att, total_word_att)
                if pmi > 0:
                    att_vec[att] = pmi
                else:
                    del att_vec[att]
        return word_att_pmi

    @staticmethod
    def calculate_norm(pmi):
        word_norm = {}
        for word, context_dic in pmi.items():
            norm = 0.0
            for context, count in context_dic.items():
                norm += context_dic[context] ** 2
            word_norm[word] = np.sqrt(norm)
        return word_norm

    def create_sparse_matrix(self):
        counts = defaultdict(Counter)
        context_counter = {}
        print('Total sequences: '+str(len(self.sentences)))
        for j, s in enumerate(self.sentences):
            #print(f'{datetime.now()}:Sentence {j}')
            context_lst = self.get_context(s, j)
            for k, word in enumerate(s):
                for i, context in enumerate(context_lst[k]):
                    if i == k and isinstance(self, SentenceParser):
                        continue
                    context_counts_for_word = counts[word]
                    context_counts_for_word[context] += 1
                    context_counter[context] = 1 if context not in context_counter \
                        else context_counter[context] + 1
        self.index_word = {v: k for k, v in self.word_index.items()}
        self.word_att = counts
        self.context_count = context_counter

    def get_similarities(self, word_att_pmi, norm):
        result = {}
        for word in WORDS:
            word = self.word_index[word]
            for other in self.word_set:
                other = self.word_index[other]
                same_labels = set(word_att_pmi[word]).intersection(word_att_pmi[other])
                mone = 0
                for att in same_labels:
                    mone += word_att_pmi[word][att] * word_att_pmi[other][att]
                result[other] = mone / (norm[word] * norm[other])
            top_pmi = {self.index_word[k]: v for k, v in sorted(word_att_pmi[word].items(),
                                                            key=lambda item: item[1], reverse=True)[0: 10]}
            top_similarity = {self.index_word[k]: v for k, v in sorted(result.items(),
                                                                       key=lambda item: item[1], reverse=True)[0: 10]}
            self.print_top(top_pmi,top_similarity, word)

    def print_top(self, pmi, sim, word):
        print(self.index_word[word] + ": " + str(pmi))
        print(self.index_word[word] + ": " + str(sim.keys()))

    def load_txt(self, vocabulary):
        sentences = []
        sen = []
        with open(vocabulary, encoding='utf8') as f:
            for line in f:
                # if we get the end of the sentence add it to all sentences
                if line == '\n':
                    sentences.append(sen)
                    sen = []
                    continue
                lemma = line.split('\t')[2]
                if line.split('\t')[4] in CONTENT_CLASSES and lemma in self.word_set:
                    sen.append(self.word_index[lemma])
        return sentences


class SentenceParser(Parser):
    def get_context(self, s, sen_index=None):
        return [s for _ in range(len(s))]


class WindowParser(Parser):
    def get_context(self, s, sen_index=None):
        return [s[i - 2:i] + s[i + 1:i + 3] for i in range(len(s))]


class ContextParser(Parser):

    def get_context(self, s, sen_index=None):
        context_sen = self.con_sentences[sen_index]
        max_index = max(self.word_index.values())+1
        context_dict = {k: [] for k in context_sen.keys()}
        for id_loc, values in context_sen.items():
            word, head_loc, feats = values
            if head_loc in context_sen:
                # taking the head
                word_head, _, feats_head = context_sen[head_loc]
                token = (word_head, feats, 'up')
                # check if context exist in dictionary and add to the head and to the token itself
                if token not in self.word_index:
                    self.word_index[token] = max_index
                    max_index += 1
                context_dict[id_loc].append(self.word_index[token])
                token_head = (word, feats, 'down')
                if token_head not in self.word_index:
                    self.word_index[token_head] = max_index
                    max_index += 1
                context_dict[head_loc].append(self.word_index[token_head])
        # self.index_word = {v: k for k, v in self.word_index.items()}
        # print({self.index_word[context_sen[k][0]]: [
        #     [self.index_word[self.index_word[i][0]], self.index_word[i][1], self.index_word[i][2]] for i in v] for k, v
        #  in context_dict.items()})
        return list(context_dict.values())

    def load_txt(self, vocabulary):
        self.con_sentences = []
        context_sen = {}
        sentences = []
        sen = []
        with open(vocabulary, encoding='utf8') as f:
            for line in f:
                # if we get the end of the sentence add it to all sentences
                if line == '\n':
                    sentences.append(sen)
                    self.con_sentences.append(context_sen)
                    context_sen = {}
                    sen = []
                    continue
                ID, _, lemma, _, CLASS, _, HEAD, FEATS, _, _ = line.split('\t')
                if CLASS in CONTENT_CLASSES and lemma in self.word_set:
                    context_sen[int(ID)] = [self.word_index[lemma], int(HEAD), FEATS]
                    sen.append(self.word_index[lemma])
            return sentences

    def print_top(self, pmi, sim, word):
        print(self.index_word[word] + ": " + ' '.join([self.index_word[k[0]]+' '+k[1]+' ' + k[2] for k, v in pmi.items()]))
        print(self.index_word[word] + ": " + str(sim.keys()))
