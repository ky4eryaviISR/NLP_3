import copy
from abc import abstractmethod
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

WORDS = ["car", "bus", "hospital", "hotel", "gun", "bomb", "fox", "table", "bowl", "guitar", "piano"]

CONTENT_CLASSES = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
                   'VBP', 'VBZ', 'WRB'}
LEMMA_MIN = 10


class Parser(object):

    def __init__(self, path, out_file):
        self.out_file = out_file
        self.word_att = {}
        self.context_count = {}
        self.lemma_cnt = None
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
        print('Total sequences: ' + str(len(self.sentences)))
        for j, s in enumerate(self.sentences):
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
            if word not in self.word_index:
                continue
            word = self.word_index[word]
            for other in self.word_set:
                other = self.word_index[other]
                same_labels = set(word_att_pmi[word]).intersection(word_att_pmi[other])
                mone = 0
                for att in same_labels:
                     mone += word_att_pmi[word][att] * word_att_pmi[other][att]
                result[other] = mone / (norm[word] * norm[other])
            top_pmi = {self.index_word[k]: v for k, v in sorted(word_att_pmi[word].items(),
                                                                key=lambda item: item[1], reverse=True)[0: 20]}
            top_similarity = {self.index_word[k]: v for k, v in sorted(result.items(),
                                                                       key=lambda item: item[1], reverse=True)[0: 20]}
            self.print_top(top_pmi, top_similarity, word)

    def print_top(self, pmi, sim, word):
        print('1-st order similarity')
        print(self.index_word[word] + ": " + ' '.join(list(pmi.keys())))
        print('2-nd order similarity')
        print(self.index_word[word] + ": " + ' '.join(list(sim.keys())))
        word = self.index_word[word]
        with open(self.out_file + '_1-nd_similarity.csv', 'a+') as file1:
            file1.write('similar contexts for:' + word + '\n')
            for w in pmi.keys():
                file1.write(w + '\n')
        with open(self.out_file + '_2-st_similarity.csv', 'a+') as file2:
            file2.write('similar contexts for:' + word + '\n')
            for context in sim.keys():
                file2.write(str(context) + '\n')

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
        i = 0
        for id_loc, values in context_sen.items():
            word, head_loc, feats, prep = values
            if head_loc == 0:
                continue
            if prep:
                continue
            h_word, h_head_loc, h_feats, prep = context_sen[head_loc]
            context = (h_word+'_'+feats+'_up')
            if context not in self.word_index:
                self.word_index[context] = max_index
                max_index += 1
            context_dict[id_loc].append(self.word_index[context])

            context2 = (word + '_' + feats + '_down')
            if context2 not in self.word_index:
                self.word_index[context2] = max_index
                max_index += 1
            context_dict[head_loc].append(self.word_index[context2])

            if prep and h_head_loc != 0:
                hh_word, hh_head_loc, hh_feats, prep = context_sen[h_head_loc]
                context3 = (word+'_'+h_feats + '_' + h_word+'_down')
                if context3 not in self.word_index:
                    self.word_index[context3] = max_index
                    max_index += 1
                context_dict[h_head_loc].append(self.word_index[context3])
                context4 = (hh_word+'_'+h_feats+'_'+h_word+'_up')
                if context4 not in self.word_index:
                    self.word_index[context4] = max_index
                    max_index += 1
                context_dict[id_loc].append(self.word_index[context4])
        # self.index_word = {v: k for k, v in self.word_index.items()}
        # print({self.index_word[context_sen[k][0]]: [
        #     [self.index_word[self.index_word[i][0]], self.index_word[i][1], self.index_word[i][2]] for i in v] for k, v
        #  in context_dict.items()})
        return [v for k, v in context_dict.items() if k in self.con_sen_index[sen_index]]

    def load_txt(self, vocabulary):
        self.con_sentences = []
        context_sen = {}
        sentences = []
        sen = []
        sen_index = []
        self.con_sen_index = []
        with open(vocabulary, encoding='utf8') as f:
            for line in f:
                # if we get the end of the sentence add it to all sentences
                if line == '\n':
                    sentences.append(sen)
                    self.con_sentences.append(context_sen)
                    self.con_sen_index.append(sen_index)
                    context_sen = {}
                    sen_index =[]
                    sen = []
                    continue
                ID, _, lemma, _, CLASS, _, HEAD, FEATS, _, _ = line.split('\t')
                context_sen[int(ID)] = [lemma, int(HEAD), FEATS, CLASS == 'IN']
                if CLASS in CONTENT_CLASSES and lemma in self.word_set:
                    sen.append(self.word_index[lemma])
                    sen_index.append(int(ID))

            return sentences
