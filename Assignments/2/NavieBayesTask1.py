import math

d = "fast couple shoot fly".split()

data = [("fun couple love love".split(), "comedy"),
        ("fast furious shoot".split(), "action"),
        ("couple fly fast fun fun".split(), "action"),
        ("furious shoot shoot fun".split(), "action"),
        ("fly fast shoot love".split(), "action")]
data_probabilities = {}
data_set = []
data_set = set([data_set.extend(x) for x, y in data])
data_set_length = len(data_set)
label
for x, y in data:
    for word in x:
        if (word, y) in data_probabilities.keys():
            data_probabilities[(word, y)] += 1
        else:
            data_probabilities[(word, y)] = 1
