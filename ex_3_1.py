from datetime import datetime
from sys import argv

from Parser import SentenceParser, WindowParser, ContextParser

params = {
    1: SentenceParser,
    2: WindowParser,
    3: ContextParser
}

if __name__ == '__main__':

    var = int(argv[2]) if len(argv) > 2 else 1 # default the first variation
    vocabulary = argv[1]

    # vocabulary = 'smallSet.txt'#argv[1]
    print(f"{datetime.now()}:Load file with parser {params[var]}")
    parser = params[var](vocabulary)
    print(f"{datetime.now()}:Create sparse matrix")
    parser.create_sparse_matrix()
    print(f"{datetime.now()}:Calculate PMI")
    word_att_pmi = parser.calculate_pmi()
    print(f"{datetime.now()}:Calculate Norm")
    norm = parser.calculate_norm(word_att_pmi)
    print(f"{datetime.now()}:Calculate Similarity")
    parser.get_similarities(word_att_pmi, norm)
    print(f"{datetime.now()}:Finish")


