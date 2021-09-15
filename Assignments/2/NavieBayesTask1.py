d = "fast couple shoot fly".split()

data = [("fun couple love love".split(), "comedy"),
        ("fast furious shoot".split(), "action"),
        ("couple fly fast fun fun".split(), "comedy"),
        ("furious shoot shoot fun".split(), "action"),
        ("fly fast shoot love".split(), "action")]


class NaiveBayesSentimentClassifier:

    def __init__(self):
        self.label_probability = {}
        self.data_probabilities = {}
        self.data_set = []
        self.data_set_length = 0

    def fit(self, features, targets):
        for xi in features:
            self.data_set.extend(xi)
        self.data_set = set(self.data_set)
        self.data_set_length = len(self.data_set)
        for x, y in zip(features, targets):
            for word in set(x):
                if (word, y) in self.data_probabilities:
                    self.data_probabilities[(word, y)] += 1
                else:
                    self.data_probabilities[(word, y)] = 1
            if y in self.label_probability:
                self.label_probability[y] += 1
            else:
                self.label_probability[y] = 1

    def predict(self, d):
        results = {}
        for label in self.label_probability:
            probability = self.label_probability[label]
            p = probability / len(data)
            for word in d:
                if word in self.data_set:
                    p_wi = ((self.data_probabilities[(word, label)] if (word, label) in self.data_probabilities else 0) + 1) / \
                           (probability + self.data_set_length)
                    p *= p_wi
            results[label] = p
        return results


sentiment_classifier = NaiveBayesSentimentClassifier()
sentiment_classifier.fit([x for x, y in data], [y for x, y in data])
print(sentiment_classifier.predict(d))
