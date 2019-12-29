import copy
from collections import defaultdict, Counter
from datetime import datetime
from sys import argv
import numpy as np

WORDS = ["car", "bus", "hospital", "hotel", "gun", "bomb", "fox", "table", "bowl", "guitar", "piano"]
CONTENT_CLASSES = {'jj', 'jjr', 'jjs', 'nn', 'nns', 'nnp', 'nnps', 'rb', 'rbr', 'rbs', 'vb', 'vbd', 'vbg', 'vbn',
                   'vbp', 'vbz', 'wrb'}

LEMMA_MIN = 100


def PMI_smooth(word, context,total):
    P_xy = word_att[word][context]/total
    P_x = sum(word_att[word].values())/total
    P_y = context_count[context]/total
    return np.log(P_xy/(P_x*P_y))


def load_txt(vocabulary):
    sentences = []
    sen = []
    lemma_count = {}
    for line in open(vocabulary, encoding='utf8').read().split('\n'):
        # if we get the end of the sentence add it to all sentences
        if line == '':
            sentences.append(sen)
            sen = []
            continue
        lemma = line.split('\t')[2]
        if line.split('\t')[4].lower() in CONTENT_CLASSES:
            sen.append(lemma)
            lemma_count[lemma] = 1 if lemma not in lemma_count \
                else lemma_count[lemma] + 1
    return sentences, lemma_count


def get_word_index_dict(lemma_cnt):

    w_set = [lemma for lemma, count in lemma_cnt.items() if count >= LEMMA_MIN]
    # index for each lemma
    w_2_i = {k: index for index, k in enumerate(w_set)}
    i_2_w = {v: k for k, v in w_2_i.items()}
    # delete lemmas which occurrences less than LEMMA_MIN
    return w_2_i, i_2_w, w_set


def create_sparse_matrix(sentences):
    counts = defaultdict(Counter)
    context_counter = {}
    for s in sentences:
        for word in s:
            index = s.index(word)
            for i, context in enumerate(s):
                if i == index:
                    continue
                context_counts_for_word = counts[word]
                context_counts_for_word[context] += 1
                context_counter[context] = 1 if context not in context_counter \
                    else context_counter[context] + 1
    return counts, context_counter


def calculate_pmi(word_att):
    total_word_att = 0
    for word in word_att.keys():
        total_word_att += sum(word_att[word].values())
    word_att_pmi = copy.deepcopy(word_att)
    for word in word_att.keys():
        att_vec = word_att_pmi[word] = {k: v for k, v in sorted(word_att[word].items(),
                                                           key=lambda item: item[1],
                                                           reverse=True)[0:100]}
        for att, count in list(att_vec.items()):
            pmi = PMI_smooth(word, att, total_word_att)
            if pmi > 0:
                att_vec[att] = pmi
            else:
                del att_vec[att]
    return word_att_pmi


def calculate_norm(pmi):
    word_norm = {}
    for word, context_dic in pmi.items():
        norm = 0.0
        for context, count in context_dic.items():
            norm += context_dic[context]**2
        word_norm[word] = np.sqrt(norm)
    return word_norm


if __name__ == '__main__':
    vocabulary = argv[1]
    # vocabulary = 'smallSet.txt'#argv[1]
    print(f"{datetime.now()}:Load file")
    sentences, lemma_cnt = load_txt(vocabulary)
    # delete lemmas which occurrences less than 75
    print(f"{datetime.now()}:Build Dictionary")
    word_index, index_word, word_set = get_word_index_dict(lemma_cnt)
    print(f"{datetime.now()}:Clean Sentences")
    sentences = [[word_index[word] for word in s if word in word_set] for s in sentences]
    print(f"{datetime.now()}:Create sparse matrix")
    word_att, context_count = create_sparse_matrix(sentences)
    print(f"{datetime.now()}:Calculate PMI")
    word_att_pmi = calculate_pmi(word_att)
    print(f"{datetime.now()}:Calculate Norm")
    norm = calculate_norm(word_att_pmi)
    print(f"{datetime.now()}:Calculate Similarity")
    result = {}
    for word in WORDS:
        word = word_index[word]
        for other in word_set:
            other = word_index[other]
            same_labels = set(word_att_pmi[word]).intersection(word_att_pmi[other])
            mone = 0
            for att in same_labels:
                mone += word_att_pmi[word][att]*word_att_pmi[other][att]
            result[other] = mone/(norm[word]*norm[other])
        top = {index_word[k]: v for k, v in sorted(word_att_pmi[word].items(),
                                                   key=lambda item: item[1], reverse=True)[0: 10]}
        top_similarity = {index_word[k]: v for k, v in sorted(result.items(),
                                                              key=lambda item: item[1], reverse=True)[0: 10]}
        print(index_word[word] + ": " + str(top))
        print(index_word[word] + ": " + str(top_similarity))

