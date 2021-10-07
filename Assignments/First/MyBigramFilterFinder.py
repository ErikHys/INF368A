import math
import nltk
from nltk.corpus import brown
from nltk.collocations import *
from tqdm import tqdm


class MyBigramFilterFinder:

    def check(self, bigram):
        for _, cl in bigram:
            result = False
            for cls in self.including:
                if cls in cl:
                    result = True
            if not result:
                return result
        return True

    def __hypothesis_test(self, collocation, confidence=2.576):
        first, second = collocation
        sample_mean = (self.finder.ngram_fd[collocation] / self.n)
        mean_of_the_dist = ((self.corpus.count(first) / self.n) * (self.corpus.count(second) / self.n))
        t = (sample_mean - mean_of_the_dist) / (math.sqrt((sample_mean * (1 - sample_mean)) / self.n))
        return t > confidence

    def __init__(self, frequency=6):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        self.corpus = brown.words()

        self.finder = BigramCollocationFinder.from_words(self.corpus)
        self.n = len(self.corpus)
        self.finder.apply_freq_filter(frequency)
        frequency_collocations = self.finder.nbest(bigram_measures.pmi, 10000)

        tagged = [nltk.pos_tag(bigram) for bigram in frequency_collocations]
        self.including = ['NN', 'JJ']
        self.noun_and_adjectives_collocations = [bigram for bigram in tagged if self.check(bigram)]

    def get_hypothesis_tested_bigrams(self):
        '''

        :return: a list of hypothesis tested bigrams
        '''
        result = []
        pbar = tqdm(total=len(self.noun_and_adjectives_collocations), desc='Hypothesis test')
        for bigram in self.noun_and_adjectives_collocations:
            if self.__hypothesis_test((bigram[0][0], bigram[1][0])):
                result.append((bigram[0][0], bigram[1][0]))
            pbar.update()
        pbar.close()
        return result

    def get_freq_and_noun_adj_filtered(self):
        '''

        :return:a list of filtered bigrams
        '''
        return self.noun_and_adjectives_collocations