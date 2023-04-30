# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import collections
from typing import List, Dict, Any


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.automaticTuning = False
        print("Legal Labels:", self.legalLabels)

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        trainingData = list(trainingData)
        self.features = list(
            set([f for datum in trainingData for f in list(datum.keys())]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels,
                          validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter 
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """

        # Count the labels in the training set
        self.count_labels = [0 for label in self.legalLabels]
        for label in trainingLabels:
            self.count_labels[label] += 1

        # Count the features for each label in the training set
        self.featureCounts = {label: util.Counter()
                              for label in self.legalLabels}
        for features, label in zip(trainingData, trainingLabels):
            self.featureCounts[label] += features

        # Store the total number of training instances
        self.dataCount = len(trainingData)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        # Log posteriors are stored for later data analysis (autograder).
        self.posteriors = []
        for datum in testData:
            logJoint = util.Counter()
            for label in self.legalLabels:
                priorProb_Labels = math.log(
                    self.count_labels[label] / self.dataCount)

                featureProb_givenLabel = 0
                for feature, value in datum.items():
                    trueCount = self.featureCounts[label][feature] + self.k
                    falseCount = self.count_labels[label] - \
                        self.featureCounts[label][feature] + self.k
                    denominator = trueCount + falseCount

                    if value:
                        featureProb_givenLabel += math.log(
                            trueCount / denominator)
                    else:
                        featureProb_givenLabel += math.log(
                            falseCount / denominator)

                logJoint[label] = priorProb_Labels + featureProb_givenLabel

            posterior = logJoint
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)

        return guesses
