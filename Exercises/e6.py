import math
import matplotlib.pyplot as plt
import copy


# Exercise 1 plot
cherry = (442, 25)
strawberry = (60, 19)

plt.plot([0, cherry[0]], [0, cherry[1]], color="coral", label="Cherry")
plt.plot([0, strawberry[0]], [0, strawberry[1]], color="lightblue", label="Strawberry")
plt.legend()
plt.title("Word vectors")
plt.xlabel("Pie")
plt.ylabel("Sugar")
plt.show()

# Exercise Third
with open("data_e6.txt", "r") as file:
    raw_docs = file.read().split('\n\n')

idf = {"Shakespeare": 0, "poet": 0, "English": 0}
for word in idf:
    for doc in raw_docs:
        if word in doc:
            idf[word] += 1

for word in idf:
    idf[word] = math.log10(len(raw_docs)/idf[word])


print(idf)
tf = {"Shakespeare": [0, 0, 0], "poet": [0, 0, 0], "English": [0, 0, 0]}
for word in tf:
    for i, doc in enumerate(raw_docs):
        a = doc.count(word)
        tf[word][i] = math.log10(a + 1)
print(tf)
for word in tf:
    a = [x*idf[word] for x in tf[word]]
    tf[word] = a
print(tf)

#Exercise 4

# PPMI
contexts = {"Shakespeare": {"poet": 0, "works": 0, "English": 0},
            "career": {"poet": 0, "works": 0, "English": 0},
            "language": {"poet": 0, "works": 0, "English": 0}}
words = ["poet", "works", "English"]

for context in contexts:
    for word in contexts[context]:
        for doc in raw_docs:
            if word in doc and context in doc:
                contexts[context][word] += 1


for context in contexts:
    su = sum([contexts[context][word] for word in contexts[context]])
    contexts[context]['Count(context)'] = su

contexts["Count(Word)"] = {}
for word in words:
    su = sum([contexts[context][word] for context in contexts if context != "Count(Word)"])
    contexts["Count(Word)"][word] = su

contexts["Count(Word)"]['Count(context)'] = 0
for context in contexts:
    if context != "Count(Word)":
        contexts["Count(Word)"]['Count(context)'] += contexts[context]['Count(context)']


def print_matrix(m, title="Term-Context Matrix"):
    print("\n", title)
    print(end='\t')
    for context in m:
        print(context, end='\t')
    print()
    for word in words:
        print(word, end='\t\t')
        for context in m:
            print(round(m[context][word], 4), end="\t")
        print()
    print('Count()', end='\t\t')
    for context in m:
        if context != "Count(Word)":
            print(round(m[context]['Count(context)'], 4), end="\t")
    print(m["Count(Word)"]['Count(context)'], end="\t")
    print()


print_matrix(contexts)

probabilities = copy.deepcopy(contexts)
for context in contexts:
    for word in contexts[context]:
        probabilities[context][word] /= contexts["Count(Word)"]['Count(context)']

print_matrix(probabilities, "Probabilities")

ppmi = copy.deepcopy(probabilities)

for context in contexts:
    for word in contexts[context]:
        prob = probabilities[context][word] / (probabilities["Count(Word)"][word] * probabilities[context]['Count(context)'])
        if prob > 0:
            log2 = math.log2(prob)
            if log2 >= 0:
                ppmi[context][word] = log2
            else:
                ppmi[context][word] = 0
        else:
            ppmi[context][word] = 0

print_matrix(ppmi, "PPMI")