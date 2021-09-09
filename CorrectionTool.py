from nltk.corpus import wordnet


class CorrectionTool:

    def __init__(self, path="1_1_2.txt"):
        with open(path, 'r') as file:
            raw = file.read()

        self.learned_collocations = [(x.split()[0].lower(), x.split()[1].lower()) for x in raw.split('\n') if x != '']
        # This is used to check if the first word of the bigram is in the library in O(1) time
        self.learned_collocations_firsts = set([x[0] for x in self.learned_collocations])

    def correct(self, first, second):
        '''
        :param first: First word of bigram
        :param second: Second word of bigram
        :return: corrected bigram if an alternative was found, else an empty string
        '''
        for synset in wordnet.synsets(first):
            for synonym in synset.lemmas():
                f_synonym = synonym.name()
                if f_synonym in self.learned_collocations_firsts:
                    second_learned = [x[1] for x in self.learned_collocations if x[0] == f_synonym]
                    for s in second_learned:
                        for second_synset in wordnet.synsets(second):
                            for second_synonym in second_synset.lemmas():
                                s_synonym = second_synonym.name()
                                if s == s_synonym and first + ' ' + second != f_synonym + ' ' + s:
                                    print('Changed', first + ' ' + second, 'to', f_synonym + ' ' + s + '!')
                                    return f_synonym + ' ' + s
        print("Found no matching collocation.")
        return ""