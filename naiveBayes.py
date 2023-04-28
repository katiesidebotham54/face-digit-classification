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

        print(len(list(trainingData)))

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
        "*** YOUR CODE HERE ***"

        # Parameter Estimation, compute the class priors: proportion of training examples that belong to each class
        # uses Counter to count number of training instances with each label, and then divides
        # by the total number of training instances to obtain the proportion of instances with each label
        priors = dict(collections.Counter(trainingLabels))
        for label in priors.keys():
            priors[label] = priors[label] / float(len(trainingLabels))
        # featureCounts hold the count of the number of times each feature value appears given a specific label
        # feature value: value of a pixel feature in a specific image would be the brightness or color intensity at that pixel location
        featureCounts = {}
        # for number 0-9
        for label in self.legalLabels:
            # create a counter at each number
            featureCounts[label] = util.Counter()
            for k in kgrid:
                # creates a counter for each k in kgrid
                featureCounts[label][k] = util.Counter()
        # iterates over each training example
        print(len(list(trainingData)))
        print(list(trainingData))
        for i in range(len(list(trainingData))):
            label = trainingLabels[i]
            # For each example, it goes through each feature index in kgrid, looks up the value of the feature in the features dictionary
            # and increments the count of that feature value in the featureCounts dictionary
            for k in self.kgrid:
                featureCounts[label][k][self.features[k]
                                        [trainingData[i][k]]] += 1
        # Calculate conditional probabilities using Laplace smoothing
        # for each label and feature index in kgrid
        # for label in self.legalLabels:
        #     for k in kgrid:
        #         for value in featureCounts[label][k].keys():
        #             print(featureCounts[label][k][value])
        # calculates the probability of each feature value given the
        # label using Laplace smoothing, and stores the result in featureCounts
        # for value in featureCounts[label][k].keys():
        #     print("value: " + str(value))
        #     count = featureCounts[label][k][value]
        #     total = float(sum(featureCounts[label][k].values()))
        #     featureCounts[label][k][value] = (
        #         count + 1.0) / (total + len(featureCounts[label][k]))
        #     print(featureCounts[label][k])

        # Update instance variables
        self.priors = priors
        self.featureCounts = featureCounts
        self.count = [a for a in self.priors]

        # Choose best value of k using held-out validationData
        # best_accuracy = 0
        # best_k = None
        # for k in kgrid:
        #     accuracy = 0
        #     for i in range(len(list(validationData))):
        #         scores = util.Counter()
        #         for label in self.legalLabels:
        #             score = self.priors[label]
        #             for j in range(len(list(validationData))[i]):
        #                 score *= self.featureCounts[label][j][self.features[j]
        #                                                       [validationData[i][j]]]
        #             scores[label] = score
        #         if validationLabels[i] == scores.argMax():
        #             accuracy += 1
        #     accuracy /= len(validationData)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_k = k

        # # Set best k value
        # self.k = best_k

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
        # datum is a dic with (feature,value) pairs
        "*** YOUR CODE HERE ***"
        # iterating for each label (a number 0-9)
        for label in self.legalLabels:
            # calculate the log prior probability of the label
            logPrior = math.log(self.priors[label])
            # iterate over each feature in the datum
            for feature, value in datum.items():
                # print(f"Feature: {feature}, Value: {value}")
                # calculate the log conditional probability of the feature given the label
                logCondProb = math.log(self.featureCounts[label][feature][value] + 1) - math.log(
                    self.count[label] + len(self.featureCounts[label]))
                # add the log conditional probability to the log joint distribution
                logJoint[label] += logCondProb
            # add the log prior probability to the log joint distribution
            # logJoint[label] += logPrior
        # return the log of the estimated probability that the data point belongs to each possible label
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2) 

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        # odds ratio: the ratio of the probability of a feature being 1 given label1
        # and the probability of the same feature being 1 given label2.
        featuresOdds = []
        # dict for storing odds ratio for each feature
        odds_ratios = {}
        for feature in self.features:
            p_f_given_label1 = (
                self.featureCounts[label1][feature] + 1) / (self.count[label1] + 2)
            p_f_given_label2 = (
                self.featureCounts[label2][feature] + 1) / (self.count[label2] + 2)
            # calc odds ratio
            odds_ratio = p_f_given_label1 / p_f_given_label2
            odds_ratios[feature] = odds_ratio

        # sort the features by odds ratio in descending order
        sorted_features = sorted(
            odds_ratios.keys(), key=lambda x: odds_ratios[x], reverse=True)

        # take the top 100 features
        featuresOdds = sorted_features[:100]
        "*** YOUR CODE HERE ***"

        return featuresOdds
