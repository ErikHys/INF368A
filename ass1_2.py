from nltk.corpus import wordnet

with open('1_1_2.txt', 'r') as file:
    raw = file.read()

learned_collocations = [(x.split()[0].lower(), x.split()[1].lower()) for x in raw.split('\n') if x != '']
learned_collocations_firsts = set([x[0] for x in learned_collocations])
learned_collocations_seconds = set([x[1] for x in learned_collocations])


def correction_tool(first, second):
    for synset in wordnet.synsets(first):
        for synonym in synset.lemmas():
            f_synonym = synonym.name()
            if f_synonym in learned_collocations_firsts:
                second_learned = [x[1] for x in learned_collocations if x[0] == f_synonym]
                for s in second_learned:
                    for second_synset in wordnet.synsets(second):
                        for second_synonym in second_synset.lemmas():
                            s_synonym = second_synonym.name()
                            if s == s_synonym and first + ' ' + second != f_synonym + ' ' + s:
                                print('Changed', first + ' ' + second, 'to', f_synonym + ' ' + s, '!')
                                return f_synonym + ' ' + s
    print("Found no collocation")


correction_tool("richly", "school")
