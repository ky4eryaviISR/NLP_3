from sys import argv
import numpy as np
from collections import defaultdict, Counter, OrderedDict


def PMI_smooth(word, context):
    P_xy = counts[word][context]/total
    P_x = count_word[word]/total
    P_y = count_word[context]/total
    return np.log(P_xy/(P_x*P_y))


def filter_words():
    lemmas = [[line.split('\t')[0]]
              for line in open(vocabulary, encoding='utf8').read().split('\n') if len(line) > 0]
def load_txt():
    sentences = []
    sen = []
    word_labels = {}
    index = 0
    for line in open(vocabulary, encoding='utf8').read().split('\n'):
        if line == '':
            sentences.append(sen)
            sen = []
            continue
        word = line.split('\t')[2]
        if word not in word_labels:
            word_labels[word] = index
            index += 1
        sen.append(word)
    labek_2_word = {v:k for k,v in word_labels.items()}
    return sentences, word_labels,labek_2_word


if __name__ == '__main__':
    vocabulary = argv[1]
    filter_words()
    sentences, word_2_label, label_2_word = load_txt()
    count_word = {}
    counts = defaultdict(Counter)
    for sen in sentences:
        for word in sen:
            count_word[word] = 1 if word not in count_word else count_word[word] + 1
        for word, context in [(word, con) for word in sen for con in sen if word != con]:
            if word =='a' and context =='the':
                print(sen)

            context_counts_for_word = counts[word]
            context_counts_for_word[context] += 1

    for word, count in count_word.items():
        if count < 75:
            print(word)
            counts.pop(word)
            continue
        counts[word] = {k: v for k,v in sorted(counts[word].items(),key=lambda item:item[1],reverse=True)[0:100]}

    top_50_words = {k: v for k,v in sorted(count_word.items(),key=lambda item: item[1],reverse=True)[:50]}
    features = [feature for k, v in counts.items() for feature in v.keys()]
    feature_dep = {feat: count for feat, count in zip(Counter(features).keys(),Counter(features).values())}
    top_50_contex = {k: v for k, v in sorted(feature_dep.items(), key=lambda item: item[1], reverse=True)[:50]}

    total = sum(Counter(features).values())
    word_map = {key: i for i, key in enumerate(counts.keys())}


    pmi_matrix = np.zeros((len(word_map), len(word_map)))
    PMI_smooth('a', 'the')
