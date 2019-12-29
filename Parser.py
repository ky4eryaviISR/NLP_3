import numpy as np

class Parser(object):

    def __init__(self):
        self.word_att = {}
        self.context_count = {}

    def PMI_smooth(self,word, context, total):
        P_xy = self.word_att[word][context] / total
        P_x = sum(self.word_att[word].values()) / total
        P_y = self.context_count[context] / total
        return np.log(P_xy / (P_x * P_y))
