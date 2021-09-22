
class NaiveBayesSentimentClassifier:

    def __init__(self):
        """
        Initialize attributes
        """
        self.vocabulary_count = {}
        self.label_count = {}
        self.data_count = {}
        self.data_set = []
        self.data_set_length = 0
        self.data_length = 0

    def length_setup_helper(self, data):
        """
        Stores the length of the of the data, and the unique length of the data.
        Only used inside fit function.
        :param data: all training documents
        """
        self.data_length = len(data)
        for xi in data:
            self.data_set.extend(xi)
        self.data_set = set(self.data_set)
        self.data_set_length = len(self.data_set)

    def count_data_helper(self, key):
        """
        Stores the Count(wj, cj) values.
        Only used inside fit function.
        :param key: (Word, label) e.g. ("fast", "action")
        """
        if key in self.data_count:
            self.data_count[key] += 1
        else:
            self.data_count[key] = 1

    def count_vocabulary_helper(self, y, word):
        """
        Used to store the total length of all documents with label y.
        Only used inside fit function.
        :param y: Label e.g. "comedy"
        :param word: e.g. "fly"
        """
        if y in self.vocabulary_count:
            self.vocabulary_count[y].append(word)
        else:
            self.vocabulary_count[y] = [y]

    def count_labels_helper(self, y):
        """
        How many times label y appears in training data
        :param y: label
        """
        if y in self.label_count:
            self.label_count[y] += 1
        else:
            self.label_count[y] = 1

    def fit(self, features, targets):
        """
        Fits the training data to the class instance.
        :param features: all documents, words needs to be tokenized
        :param targets: labels
        """
        self.length_setup_helper(features)

        for x, y in zip(features, targets):
            for word in set(x):

                self.count_data_helper((word, y))

                self.count_vocabulary_helper(y, word)

            self.count_labels_helper(y)

        for label in self.vocabulary_count.keys():
            self.vocabulary_count[label] = len(self.vocabulary_count[label])

    def predict(self, d):
        """
        Predicts the class label of document d
        :param d: a list of words
        :return: a tuple containing the predicted class and a dictionary with all class values
        """
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
    """
    Example with the data from task 1 assignment 2
    """
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
    """
    Example with data from lecture 4 slide 46
    """
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

