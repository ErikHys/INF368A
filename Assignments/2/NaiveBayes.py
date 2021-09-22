
class NaiveBayesSentimentClassifier:

    def __init__(self):
        self.vocabulary_count = {}
        self.label_count = {}
        self.data_count = {}
        self.data_set = []
        self.data_set_length = 0
        self.data_length = 0

    def length_setup_helper(self, data):
        self.data_length = len(data)
        for xi in data:
            self.data_set.extend(xi)
        self.data_set = set(self.data_set)
        self.data_set_length = len(self.data_set)

    def count_data_helper(self, key):
        if key in self.data_count:
            self.data_count[key] += 1
        else:
            self.data_count[key] = 1

    def count_vocabulary_helper(self, y, word):
        if y in self.vocabulary_count:
            self.vocabulary_count[y].append(word)
        else:
            self.vocabulary_count[y] = [y]

    def count_labels_helper(self, y):
        if y in self.label_count:
            self.label_count[y] += 1
        else:
            self.label_count[y] = 1

    def fit(self, features, targets):
        self.length_setup_helper(features)

        for x, y in zip(features, targets):
            for word in set(x):

                self.count_data_helper((word, y))

                self.count_vocabulary_helper(y, word)

            self.count_labels_helper(y)

        for label in self.vocabulary_count.keys():
            self.vocabulary_count[label] = len(self.vocabulary_count[label])

    def predict(self, d):
        results = {}
        for label in self.label_count:
            probability = self.label_count[label]
            p = probability / self.data_length
            for word in d:
                if word in self.data_set:
                    p_wi = ((self.data_count[(word, label)] if (word, label) in self.data_count else 0) + 1) / \
                           (self.vocabulary_count[label] + self.data_set_length)
                    p *= p_wi
            results[label] = p
        return max(zip(results.values(), results.keys()))[1], results


def run_example():
    d = "fast couple shoot fly".split()

    data = [("fun couple love love".split(), "comedy"),
            ("fast furious shoot".split(), "action"),
            ("couple fly fast fun fun".split(), "comedy"),
            ("furious shoot shoot fun".split(), "action"),
            ("fly fast shoot love".split(), "action")]
    sentiment_classifier = NaiveBayesSentimentClassifier()
    x, y = [x for x, y in data], [y for x, y in data]
    sentiment_classifier.fit(x, y)
    result = sentiment_classifier.predict(d)
    print("Predicted class:", result[0], "Values:", result[1])


def run_lecture_example():
    d = "predictable with no fun".split()

    data = [("just plain boring".split(), "-"),
            ("entirely predictable and lacks energy".split(), "-"),
            ("no surprises and very few laughs".split(), "-"),
            ("very powerful".split(), "+"),
            ("the most fun film of the summer".split(), "+")]

    sentiment_classifier = NaiveBayesSentimentClassifier()
    x, y = [x for x, y in data], [y for x, y in data]
    sentiment_classifier.fit(x, y)
    result = sentiment_classifier.predict(d)
    print("Predicted class:", result[0], "Values:", result[1])

run_lecture_example()