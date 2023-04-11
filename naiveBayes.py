# naiveBayes.py

import util
import classificationMethod
import math
import collections


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
        self.count = None
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

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(
            set([f for datum in trainingData for f in datum.keys()]))

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

        Estimate conditional probabilities from the training data for each possible value of k given in the list kgrid.
        """

        # Compute the class priors: proportion of training examples that belong to each class
        priors = dict(collections.Counter(trainingLabels))
        for label in priors.keys():
            priors[label] = priors[label] / float(len(trainingLabels))

        # Initialize featureCounts with Laplace smoothing
        featureCounts = {}
        for label in self.legalLabels:
            featureCounts[label] = util.Counter()
            for k in self.kgrid:
                featureCounts[label][k] = util.Counter()

        # Collect counts for each label and feature
        for i in range(len(trainingData)):
            label = trainingLabels[i]
            for k in self.kgrid:
                featureCounts[label][k][trainingData[i][k]] += 1

        # Calculate probabilities using Laplace smoothing
        for label in self.legalLabels:
            for k in self.kgrid:
                for value in featureCounts[label][k].keys():
                    count = featureCounts[label][k][value]
                    total = float(sum(featureCounts[label][k].values()))
                    featureCounts[label][k][value] = (
                        count + 1.0) / (total + len(featureCounts[label][k]))

        # Update instance variables
        self.priors = priors
        self.featureCounts = featureCounts
        self.count = [a for a in self.priors]

        # Choose best value of k using held-out validationData
        best_accuracy = 0
        best_k = None
        for k in self.kgrid:
            accuracy = 0
            for i in range(len(validationData)):
                scores = util.Counter()
                for label in self.legalLabels:
                    score = self.priors[label]
                    for j in range(len(validationData[i])):
                        score *= self.featureCounts[label][j][validationData[i][j]]
                    scores[label] = score
                if validationLabels[i] == scores.argMax():
                    accuracy += 1
            accuracy /= len(validationData)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        # Set best k value
        self.k = best_k

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        # Log posteriors are stored for later data analysis (autograder).
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.    
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2) 

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
