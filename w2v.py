import numpy as np
import sys
import time

start = time.time()

words_file = sys.argv[1]
context_file = sys.argv[2]
out_file = sys.argv[3]
target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar',
                'piano']

word_to_id = {}
words_vocabulary = []

word_vectors = []
target_vecs = {}

i = 0

with open(words_file, 'r', encoding='utf-8') as file:
    for line in file:
        i += 1
        line = line.split()
        word, vec = line[0], np.array(line[1:], dtype=float)
        word_to_id[word] = i-1
        if word in target_words:
            target_vecs[word] = vec
        norm = 0
        for h in vec:
            norm += h ** 2
        norm = np.sqrt(norm)
        norm_vec = vec / norm

        words_vocabulary.append(word)
        word_vectors.append(vec)

    word_vectors = np.array(word_vectors)
    words_vocabulary = np.array(words_vocabulary)
    W_word = np.array([vec for vec in word_vectors])


context_to_id = {}
contexts = []

context_vecs = []

i = 0
with open(context_file, 'r', encoding='utf-8') as file:
    for line in file:
        i += 1
        line = line.split()
        con, vec = line[0], np.array(line[1:], dtype=float)
        context_to_id[con] = i
        contexts.append(con)
        context_vecs.append(vec)

    context_vecs = np.array(context_vecs)
    contexts = np.array(contexts)

    W_ctx = np.array([vec for vec in context_vecs])


def k_sim(word, k=20):
    word_vec = W_word[word_to_id[word]]
    similarities = W_word.dot(word_vec)
    sims = similarities.argsort()[-1:-k-1:-1]
    sims = np.array(sims)
    similar_words = np.array(words_vocabulary)[sims]
    return similar_words


def k_context(word, k=10):
    word_vec = target_vecs[word]
    similarities = W_ctx.dot(word_vec)
    sims = similarities.argsort()[-1:-k-1:-1]
    sims = np.array(sims)
    similar_words = np.array(contexts)[sims]
    return similar_words


if __name__ == '__main__':

    for word in target_words:
        print(word)
        sim_word = k_sim(word)
        sim_context = k_context(word)
        print(sim_word)
        print(sim_context)
        with open(out_file + '_2-nd_similarity.csv', 'w') as file1:
            file1.write('similar contexts for:' + word + '\n')
            for w in sim_word:
                file1.write(w + '\n')
        with open(out_file + '_1-st_similarity.csv', 'w') as file2:
            file2.write('similar contexts for:' + word + '\n')
            for context in sim_context:
                file2.write(str(context) + '\n')

end = time.time() - start

print("time", end)
