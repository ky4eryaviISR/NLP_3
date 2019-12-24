from sys import argv
import numpy as np
from collections import defaultdict, Counter, OrderedDict

LEMMA_MIN = 10

def PMI_smooth(word, context):
    P_xy = word_att_counter[word][context]/total_word_att
    P_x = sum(word_att_counter[word].values())/total_word_att
    P_y = att_cnt[context]/total_word_att
    print(P_xy/(P_x*P_y))
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
        if line.split('\t')[7] in ['p', 'adpmod', 'det', 'cc', 'adpcomp', 'aux', 'auxpass','nsubj','mark','adp','neg']\
                or line.split('\t')[2] in ['.', 'a', ')', '(']:
            continue
        line_dict = {
            'id': line.split('\t')[0],
            'lemma': line.split('\t')[2],
            'tag': line.split('\t')[3],
            'feat': line.split('\t')[5]
        }
        sen.append(line_dict)
        lemma = line_dict['lemma']
        lemma_count[lemma] = 1 if lemma not in lemma_count \
            else lemma_count[lemma] + 1
    return sentences, lemma_count


def get_word_counter():
    counts = defaultdict(Counter)
    context_counter = {}
    for s in sentences:
        permutation = [(word['lemma'], con['lemma']) for word in s for con in s]
        for word, context in permutation:
            context_counts_for_word = counts[word]
            context_counts_for_word[context] += 1
            context_counter[context] = 1 if context not in context_counter \
                else context_counter[context]+1
    return counts, context_counter


if __name__ == '__main__':
    vocabulary = argv[1]
    # loading the sentences and count for each lemma
    sentences, lemma_count = load_txt()
    # get count vector for each word and attr
    word_att_counter, att_cnt = get_word_counter()

    # delete lemmas which occurrences less than 75
    word_set = [lemma for lemma, count in lemma_count.items() if count > LEMMA_MIN]
    # get 100 common attributes/context
    att_cnt_set = {k: v for k, v in sorted(att_cnt.items(),
                                           key=lambda item: item[1],
                                           reverse=True)[0:100]}
    # index for each lemma
    lemma_2_index = {k: index for index, k in enumerate(word_set)}
    index_2_lemma = {v: k for k, v in lemma_2_index.items()}
    # index for each attribute/context
    att_2_index = {k: index for index, k in enumerate(att_cnt_set)}
    index_2_att = {v: k for k, v in att_2_index.items()}

    pmi_matrix = {w: {} for w in word_set}
    total_word_att = sum([c for cont in word_att_counter.values() for c in cont.values()])
    for w in word_set:
        for att in att_cnt_set:
            if att not in word_att_counter[w]:
                continue
            pmi = PMI_smooth(w, att)
            if pmi <= 0:
                continue
            # create sparse matrix
            att_id = att_2_index[att]
            pmi_matrix[w][att_id] = pmi





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
