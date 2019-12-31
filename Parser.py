import copy
from abc import abstractmethod
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

WORDS = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']

LEMMA_MIN = 100
CONTENT_POS = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB',
               'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB'}
FUNC_WORDS = {'be', 'is', 'am', 'are', 'have', 'got', 'do',
              'not', 'he', 'they', 'anybody', 'it', 'one',
              'when', 'while', ']'}


class Parser(object):

    def __init__(self, path):
        self.most_sim = {}
        self.context_count = {}
        self.word_att = {}
        self.lemma_cnt = None
        self.sen_cnt = 0
        print(f"{datetime.now()}:File Parsing")
        self.sentences = self.load_txt(path)
        self.contexts = self.create_contexts(self.sentences)

    def PMI_smooth(self, word, context, total):
        P_xy = self.word_att[word][context] / total
        P_x = sum(self.word_att[word].values()) / total
        P_y = self.context_count[context] / total
        return np.log(P_xy / (P_x * P_y))

    def create_contexts(self, sentences):
        c = []
        for i, s in enumerate(sentences):
            c.append(self.get_context(s, sen_index=i))
            if i % 100000 == 0:
                print(f"{datetime.now()}: Pass 100,000 sentences")
        return c

    @abstractmethod
    def get_context(self, s, sen_index=None):
        pass

    def calculate_pmi(self):

        total_word_att = 0
        for word in self.word_att.keys():
            total_word_att += sum(self.word_att[word].values())
        word_att_pmi = copy.deepcopy(self.word_att)
        for word in self.word_att.keys():
            att_vec = word_att_pmi[word] = {k: v for k, v in sorted(self.word_att[word].items(),
                                                                    key=lambda item: item[1],
                                                                    reverse=True)[0:50]}
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
        print('Total sequences: ' + str(len(self.sentences)))
        for j, s in enumerate(self.sentences):
            if j % 100000 == 0:
                print(f"{datetime.now()}: Passed 100,000")
            context_lst = self.contexts[j]
            for k, word in enumerate(s):
                if self.lemma_cnt[word] < LEMMA_MIN:
                    continue
                for i, context in enumerate(context_lst[k]):
                    if self.context_count[context] > 100:
                        if i == k and isinstance(self, SentenceParser):
                            continue
                        context_counts_for_word = counts[word]
                        context_counts_for_word[context] += 1

        self.word_att = counts
        if isinstance(self, ContextParser):
            with open('counts_contexts_dep.txt', 'w') as f:
                for c, cnt in {k: v for k, v in
                               sorted(self.context_count.items(),
                                      key=lambda item: item[1],
                                      reverse=True)[0:50]}.items():
                    f.write(f"{c} {cnt}\n")
                with open('counts_words.txt', 'w') as f:
                    for c, cnt in {k: v for k, v in
                                   sorted(self.lemma_cnt.items(),
                                          key=lambda item: item[1],
                                          reverse=True)[0:50]}.items():
                        f.write(f"{c} {cnt}\n")

    def get_similarities(self, word_att_pmi, norm):
        word_set = [k for k,v in self.lemma_cnt.items() if v > LEMMA_MIN]
        result = {}
        for word in WORDS:
            for other in word_set:
                same_labels = set(word_att_pmi[word]).intersection(word_att_pmi[other])
                mone = 0
                for att in same_labels:
                     mone += word_att_pmi[word][att] * word_att_pmi[other][att]
                if len(same_labels) > 0:
                    result[other] = mone / (norm[word] * norm[other])
                else:
                    result[other] = 0
            top_pmi = {k: v for k, v in sorted(word_att_pmi[word].items(),
                                               key=lambda item: item[1], reverse=True)[0: 20]}
            top_similarity = {k: v for k, v in sorted(result.items(),
                                                      key=lambda item: item[1], reverse=True)[0: 20]}
            self.print_top(top_pmi, top_similarity, word)
            self.most_sim[word] = list(top_similarity.keys())

    def print_top(self, pmi, sim, word):
        print('1-st order similarity')
        print(word + ": " + ' '.join(list(pmi.keys())))
        print('2-nd order similarity')
        print(word + ": " + ' '.join(list(sim.keys())))
        with open(self.get_class() + '_1-nd_similarity.csv', 'a+',encoding='utf-8') as file1:
            file1.write('similar contexts for:' + word + '\n')
            for w in pmi.keys():
                file1.write(w + '\n')
        with open(self.get_class() + '_2-st_similarity.csv', 'a+',encoding='utf-8') as file2:
            file2.write('similar contexts for:' + word + '\n')
            for context in sim.keys():
                file2.write(str(context) + '\n')

    @abstractmethod
    def get_class(self):
        pass

    def load_txt(self, vocabulary):
        lemma_count = {}
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
                if line.split('\t')[4] in CONTENT_POS and lemma not in FUNC_WORDS:
                    sen.append(lemma)
                    lemma_count[lemma] = 1 if lemma not in lemma_count \
                        else lemma_count[lemma] + 1
            self.lemma_cnt = lemma_count
        return sentences


class SentenceParser(Parser):
    def get_context(self, s, sen_index=None):
        con = [s for _ in range(len(s))]
        for row in con:
            for w in row:
                self.context_count[w] = 1 if w not in self.context_count \
                    else self.context_count[w] + 1
        return con

    def get_class(self):
        return "SentenceParser"


class WindowParser(Parser):
    def get_context(self, s, sen_index=None):
        con = [s[i - 2:i] + s[i + 1:i + 3] for i in range(len(s))]
        for row in con:
            for w in row:
                self.context_count[w] = 1 if w not in self.context_count \
                    else self.context_count[w] + 1
        return con

    def get_class(self):
        return "WindowParser"


class ContextParser(Parser):

    def get_class(self):
        return "ContextParser"

    def get_context(self, s, sen_index=None):
        context_sen = self.con_sentences[sen_index]
        context_dict = {k: [] for k in context_sen.keys()}
        for id_loc, values in context_sen.items():
            word, head_loc, feats, prep = values
            if head_loc == 0 or prep or head_loc not in context_sen:
                continue
            h_word, h_head_loc, h_feats, h_prep = context_sen[head_loc]
            if h_word not in self.lemma_cnt:
                continue
            if not h_prep:
                context = (h_word + '_' + feats + '_up')
                self.add_context(context)
                context_dict[id_loc].append(context)
                context2 = (word + '_' + feats + '_down')
                context_dict[head_loc].append(context2)
                self.add_context(context2)
            if h_prep and h_head_loc != 0 and h_head_loc in context_sen:
                hh_word, hh_head_loc, hh_feats, prep = context_sen[h_head_loc]
                context3 = (word + '_' + h_feats + '_' + h_word+'_down')
                self.add_context(context3)
                context_dict[h_head_loc].append(context3)
                context4 = (hh_word+'_'+h_feats+'_'+h_word+'_up')
                context_dict[id_loc].append(context4)
                self.add_context(context4)

        return [v for k, v in context_dict.items() if k in self.con_sen_index[sen_index]]

    def add_context(self,context):
        self.context_count[context] = 1 if context not in self.context_count \
            else self.context_count[context] + 1

    def load_txt(self, vocabulary):
        self.con_sentences = []
        context_sen = {}
        sentences = []
        sen = []
        sen_index = []
        lemma_count = {}
        self.con_sen_index = [None]*self.sen_cnt
        i = 0
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
                    i += 1
                    continue
                ID, _, lemma, _, CLASS, _, HEAD, FEATS, _, _ = line.split('\t')
                if CLASS in CONTENT_POS and lemma not in FUNC_WORDS:
                    context_sen[int(ID)] = [lemma, int(HEAD), FEATS, False]
                    sen.append(lemma)
                    sen_index.append(int(ID))
                elif CLASS == 'IN':
                    context_sen[int(ID)] = [lemma, int(HEAD), FEATS, True]
                if CLASS in CONTENT_POS and lemma not in FUNC_WORDS:
                    lemma_count[lemma] = 1 if lemma not in lemma_count \
                        else lemma_count[lemma] + 1
            self.lemma_cnt = lemma_count
            print(f"{datetime.now()}: Start creating contexes")
            return sentences


