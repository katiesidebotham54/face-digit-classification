# knn.py
import util
import math
import statistics
from numpy.linalg import norm
from collections import defaultdict


class knnClassifier:
    """
    k-Nearest Neighbor classifier.
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.k = 10
        self.max_iterations = max_iterations
        self.weights = {}

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Preprocess the training data and save it for later use.
        """
        self.trainingData = list(trainingData)
        self.trainingLabels = list(trainingLabels)
        validationData = list(validationData)
        validationLabels = list(validationLabels)
        self.features = list(
            set([f for datum in trainingData for f in datum.keys()]))

    def getDistance(self, test_datum, train_datum):
        """
            Calculate the Euclidean distance between two data points.
        """
        # distance = 0
        # for feature in self.features:
        #     distance += abs(test_datum[feature] - train_datum[feature])
        # return distance
        return norm(test_datum-train_datum)

    def classify(self, testData):
        """
        Find the k closest 'neighbors' of the test image in the training data
        and then return the label which appeared the most. If there is a tie
        then pick the label of the training image with the lowest distance.
        """
        guesses = []
        for datum in testData:
            distances = []
            for i in range(len(self.trainingData)):
                distances.append(
                    (self.trainingLabels[i], self.getDistance(datum, self.trainingData[i])))
            distances.sort(key=lambda x: x[1])
            k_nearest = distances[:self.k]
            counter = util.Counter()
            for label, _ in k_nearest:
                counter[label] += 1
            guesses.append(counter.argMax())
        return guesses
