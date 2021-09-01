import nltk
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()

finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))

print(finder.nbest(bigram_measures.pmi, 10))


finder.apply_freq_filter(3)
print(finder.nbest(bigram_measures.pmi, 10))
