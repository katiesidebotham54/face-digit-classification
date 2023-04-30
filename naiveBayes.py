# naiveBayes.py

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
        self.priors = None
        self.count = util.Counter()
        self.featureCounts = None

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
        trainingData = list(trainingData)
        trainingLabels = list(trainingLabels)
        validationData = list(validationData)
        # this is a list of all features in the training set.
        self.features = list(
            set([f for datum in trainingData for f in datum.keys()]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels,
                          validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData: List[Dict], trainingLabels: List[Any], validationData: List[Any], validationLabels, kgrid: List[float]):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters. The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.

        Estimate conditional probabilities from the training data for each possible value of k given in the list kgrid.
        """
        # Compute the class priors: proportion of training examples that belong to each class
        # Use this var to get P(label)
        priors = dict(collections.Counter(trainingLabels))
        for label in priors.keys():
            priors[label] = priors[label] / float(len(trainingLabels))

        self.featureCounts = {}
        for label in self.legalLabels:
            self.featureCounts[label] = {}
            for feature in self.features:
                self.featureCounts[label][feature] = util.Counter()

        for i in range(len(trainingData)):
            label = trainingLabels[i]
            for feature, feature_value in trainingData[i].items():
                self.featureCounts[label][feature][feature_value] += 1

        best_k = None
        best_accuracy = -1

        for k in kgrid:
            self.k = k
            correct = 0
            guesses = []
            for i in range(len(validationData)):
                "*** YOUR CODE HERE ***"

                # Compute the log-joint probabilities for each label given the datum
                logJoint = self.calculateLogJointProbabilities(
                    validationData[i])
                # Find the label with the highest log-joint probability
                guessedLabel = logJoint.argMax()

                # Add the guessed label to the list of guesses
                guesses.append(guessedLabel)

                # Check if the guessed label matches the correct label
                if guessedLabel == validationLabels[i]:
                    correct += 1

            # Compute the accuracy of the classifier for this value of k
            accuracy = correct / float(len(validationLabels))

            # Update the best k value if this value of k gives a higher accuracy
            if accuracy > best_accuracy:
                best_k = k
                best_accuracy = accuracy

        # Set the k value to the best value found during tuning
        self.k = best_k
        # Update instance variables
        self.priors = priors
        print("made it here")
        self.count = [a for a in self.priors]

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        # Log posteriors are stored for later data analysis (autograder).
        posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        print("made it here")
        logJoint = util.Counter()
        # iterating for each label (a number 0-9)
        for label in self.legalLabels:
            logPrior = math.log(self.priors[label])
            # each feature is a pixel
            for feature, feature_value in datum.items():
                feature_index = self.features.index(feature)
                if (label, feature_index) in self.featureCounts and feature_value in self.featureCounts[label][feature_index]:
                    logCondProb = math.log(self.featureCounts[label][feature_index][feature_value] + 1) - math.log(
                        self.count[label] + len(self.featureCounts[label]))
                    logJoint[label] += logCondProb
                else:
                    logCondProb = math.log(
                        1) - math.log(self.count[label] + len(self.featureCounts[label]))
                    logJoint[label] += logCondProb
            logJoint[label] += logPrior
        print(logJoint)
        return logJoint
