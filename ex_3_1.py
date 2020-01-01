from datetime import datetime
from sys import argv

from Parser import SentenceParser, WindowParser, ContextParser, WORDS

params = {
    1: SentenceParser,
    2: WindowParser,
    3: ContextParser
}

if __name__ == '__main__':

    vocabulary = argv[1]
    # vocabulary = 'smallSet.txt'
    word_sim = {}


    for i, var in params.items():
        print(f"{datetime.now()}:Load file with parser {var}")
        parser = var(vocabulary)
        print(f"{datetime.now()}:Create sparse matrix")
        parser.create_sparse_matrix()
        print(f"{datetime.now()}:Calculate PMI")
        word_att_pmi = parser.calculate_pmi()
        print(f"{datetime.now()}:Calculate Norm")
        norm = parser.calculate_norm(word_att_pmi)
        print(f"{datetime.now()}:Calculate Similarity")
        parser.get_similarities(word_att_pmi, norm)
        print(f"{datetime.now()}:Finish")
        word_sim[parser.get_class()] = parser.most_sim
    with open('top20.txt', 'w') as f:
        for w in WORDS:
            f.write(w+'\n')
            for i in range(20):
                f.write(f"{word_sim['WindowParser'][w][i]} "
                        f"{word_sim['SentenceParser'][w][i]} "
                        f"{word_sim['ContextParser'][w][i]}\n")
            f.write('**********\n')



