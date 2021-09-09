import math

import nltk
from nltk.corpus import brown
from nltk.collocations import *
from tqdm import tqdm

bigram_measures = nltk.collocations.BigramAssocMeasures()
corpus = brown.words()

finder = BigramCollocationFinder.from_words(corpus)
collocations = finder.nbest(bigram_measures.pmi, 10000)

# 1.1.1
finder.apply_freq_filter(6)
frequency_collocations = finder.nbest(bigram_measures.pmi, 10000)


tagged = [nltk.pos_tag(bigram) for bigram in frequency_collocations]
including = ['NN', 'JJ']


def check(bigram):
    for _, cl in bigram:
        result = False
        for cls in including:
            if cls in cl:
                result = True
        if not result:
            return result
    return True


noun_and_adjectives_collocations = [bigram for bigram in tagged if check(bigram)]
#
# with open("1_1_1.txt", 'w') as file:
#     for bigram in noun_and_adjectives_collocations:
#         file.write(bigram[0][0] + ' ' + bigram[1][0] + '\n')


# 1.1.2

n = len(corpus)


def hypothesis_test(collocation, confidence=2.576):
    first, second = collocation
    f_count = corpus.count(first)
    s_count = corpus.count(second)
    sample_mean = (finder.ngram_fd[collocation] / n)
    mean_of_the_dist = ((f_count / n) * (s_count / n))
    t = (sample_mean - mean_of_the_dist) / (math.sqrt((sample_mean * (1 - sample_mean))/n))
    print(t)
    return t > confidence


with open("1_1_2.txt", 'w') as file:
    ts = []
    pbar = tqdm(total=len(noun_and_adjectives_collocations), desc='Hypothesis test')
    for bigram in noun_and_adjectives_collocations:
        if hypothesis_test((bigram[0][0], bigram[1][0])):
            file.write(bigram[0][0] + ' ' + bigram[1][0] + '\n')
        pbar.update()
    pbar.close()