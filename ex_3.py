from sys import argv
from collections import defaultdict, Counter, OrderedDict

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

    counts = defaultdict(Counter)
    for sen in sentences:
        for word, context in [(word, con) for word in sen for con in sen if word != con]:
            context_counts_for_word = counts[word]
            context_counts_for_word[context] += 1

    for word, counters in counts.items():
        if sum(counters) < 75:
            counts.pop(word)
        counts[word] = {k: v for k,v in sorted(counts[word].items(),key=lambda item:item[1],reverse=True)[0:100]}
    top_50_words = {k: sum(v.values()) for k,v in sorted(counts.items(),key=lambda item: sum(item[1].values()),reverse=True)[:50]}
    print('x')