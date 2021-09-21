d = "fast couple shoot fly".split()

data = [("fun couple love love".split(), "comedy"),
        ("fast furious shoot".split(), "action"),
        ("couple fly fast fun fun".split(), "comedy"),
        ("furious shoot shoot fun".split(), "action"),
        ("fly fast shoot love".split(), "action")]


class NaiveBayesSentimentClassifier:

    def __init__(self):
        self.vocabulary_count = {}
        self.label_count = {}
        self.data_count = {}
        self.data_set = []
        self.data_set_length = 0

    def fit(self, features, targets):
        for xi in features:
            self.data_set.extend(xi)
        self.data_set = set(self.data_set)
        self.data_set_length = len(self.data_set)
        for x, y in zip(features, targets):
            for word in set(x):
                if (word, y) in self.data_count:
                    self.data_count[(word, y)] += 1
                else:
                    self.data_count[(word, y)] = 1
                if y in self.vocabulary_count:
                    self.vocabulary_count[y].append(word)
                else:
                    self.vocabulary_count[y] = [y]
            if y in self.label_count:
                self.label_count[y] += 1
            else:
                self.label_count[y] = 1
        for label in self.vocabulary_count.keys():
            self.vocabulary_count[label] = len(self.vocabulary_count[label])

    def predict(self, d):
        results = {}
        for label in self.label_count:
            probability = self.label_count[label]
            p = probability / len(data)
            for word in d:
                if word in self.data_set:
                    p_wi = ((self.data_count[(word, label)] if (word, label) in self.data_count else 0) + 1) / \
                           (self.vocabulary_count[label] + self.data_set_length)
                    p *= p_wi
            results[label] = p
        return max(zip(results.values(), results.keys()))[1], results


sentiment_classifier = NaiveBayesSentimentClassifier()
x, y = [x for x, y in data], [y for x, y in data]
sentiment_classifier.fit(x, y)
result = sentiment_classifier.predict(d)
print(result)

