import nltk

sentence = "In Dusseldorf I took my hat off. But I can’t put it back on"

tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)



