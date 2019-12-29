from datetime import datetime
from sys import argv
import numpy as np
from collections import defaultdict, Counter, OrderedDict
import math

LEMMA_MIN = 5
MAX_COMMON_ATT = 100
CO_THRESHOLD = 1
target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl' , 'guitar', 'piano']
FUNCT_WORDS = [',', '.', 'the', 'a', '(', ')', 'this', 'those']
def PMI_smooth(word, context):
    P_xy = word_att_counter[word][context]/total_word_att
    P_x = sum(word_att_counter[word].values())/total_word_att
    P_y = att_cnt[context]/total_word_att
    #print(P_xy/(P_x*P_y))
    return np.log(P_xy/(P_x*P_y))


def load_txt():
    sentences = []
    sen = []
    lemma_count = {}
    for line in open(vocabulary, encoding='utf8').read().split('\n'):
        # if we get the end of the sentence add it to all sentences
        if line == '':
            sentences.append(sen)
            sen = []
            continue
        # if line.split('\t')[2] in FUNCT_WORDS:
        #     continue
        lemma = line.split('\t')[2]
        sen.append(lemma)
        lemma_count[lemma] = 1 if lemma not in lemma_count \
            else lemma_count[lemma] + 1
    return sentences, lemma_count


def get_word_counter(word_count):
    counts = defaultdict(Counter)
    context_counter = {}
    for s in sentences:
        for word in s:
            if word_count[word] >= LEMMA_MIN:
                index = s.index(word)
                for i, context in enumerate(s):
                    if i == index:
                        continue
                    context_counts_for_word = counts[word]
                    context_counts_for_word[context] += 1
                    context_counter[context] = 1 if context not in context_counter \
                        else context_counter[context]+1
    return counts, context_counter


def get_index_4_lemma(lemma_cnt):
    w_set = [lemma for lemma, cnt in lemma_cnt.items()]
    # index for each lemma
    l_2_i = {k: index for index, k in enumerate(w_set)}
    i_2_l = {v: k for k, v in l_2_i.items()}
    # delete lemmas which occurrences less than LEMMA_MIN
    w_set = [lemma for lemma, cnt in lemma_cnt.items() if cnt >= LEMMA_MIN]
    return l_2_i, i_2_l, w_set


def get_att_4_index(att_cnt):
    # index for each attribute/context
    a_2_i = {k: index for index, k in enumerate(att_cnt.keys())}
    i_2_a = {v: k for k, v in a_2_i.items()}
    return a_2_i, i_2_a


if __name__ == '__main__':
    vocabulary = argv[1]
    # loading the sentences and count for each lemma
    print(f"{datetime.now()}:Load text")
    sentences, lemma_count = load_txt()
    # get count vector for each word and attr
    print(f"{datetime.now()}:Get word counter")
    word_att_counter, att_cnt = get_word_counter(lemma_count)

    lemma_2_index, index_2_lemma, word_set = get_index_4_lemma(lemma_count)

    pmi_matrix = {w: {} for w in word_set}
    total_word_att = 0
    for word in word_att_counter.keys():
        total_word_att += sum(word_att_counter[word].values())
    print(f"{datetime.now()}:Start calculating PMI")
    word_att_set = {}
    for w in word_set:
        word_att_set[w] = [k for k, v in sorted(word_att_counter[w].items(),
                                 key=lambda item: item[1],
                                 reverse=True)][0:100]
        for att in word_att_set[w]:
            if att not in word_att_counter[w]:
                continue
            pmi = PMI_smooth(w, att)
            if pmi <= 0:
                continue
            # create sparse matrix
            att_id = lemma_2_index[att]
            pmi_matrix[w][att_id] = pmi

    word_norm = {}
    print(f"{datetime.now()}:Start calculating norm")
    for word, context_dic in word_att_counter.items():
        norm = 0.0
        for context, count in context_dic.items():
            norm += context_dic[context]**2
        word_norm[word] = math.sqrt(norm)
    print(f"{datetime.now()}:Start calculating similarity")
    for word in ['dog']:
        words = np.zeros(len(lemma_2_index))
        for att, pmi1 in pmi_matrix[word].items():
            for word2 in word_set:
                if att in pmi_matrix[word2]:
                    words[lemma_2_index[word2]] += pmi1 * pmi_matrix[word2][att]

        for i, word2 in enumerate(word_set):
            words[i] = words[i] / (word_norm[word] * word_norm[word2])
        print('x')




    # word_att_counter[word] = {k: v for k,v in sorted(word_att_counter[word].items(),
    #                                                  key=lambda item:item[1],reverse=True)[0:100]}

    # features = [feature for k, v in counts.items() for feature in v.keys()]
    # feature_dep = {feat: count for feat, count in zip(Counter(features).keys(),Counter(features).values())}
    #
    # # count 50 most common words and print it to file
    # top_50_words = {k: v for k, v in sorted(count_word.items(), key=lambda item: item[1], reverse=True)[:50]}
    # # count 50 most common context words and print it to file
    # top_50_context = {k: v for k, v in sorted(feature_dep.items(), key=lambda item: item[1], reverse=True)[:50]}
    #
    # #total = sum(Counter(features).values())
    # word_map = {key: i for i, key in enumerate(counts.keys())}


    # pmi_matrix = np.zeros((len(word_map), len(word_map)))
    # PMI_smooth('a', 'the')
