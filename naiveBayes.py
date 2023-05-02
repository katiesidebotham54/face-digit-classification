# naiveBayes.py
# -------------
import util
import classificationMethod
import math


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    NB Classifier assumes that the features in the input data are conditionally independent given the label.
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 0.1  # this is the smoothing parameter, ** use it in your train method **
        # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.automaticTuning = False

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Sets the features based on trainingData
        """
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
        # Used to count the number of times each label appears in the training set
        self.label_count = util.Counter()
        self.dataCount = len(trainingLabels)

        for label in trainingLabels:
            self.label_count[label] += 1

        # Initialize dictionary to keep track of number of times each feature appears in each label
        self.featureCounts = {}
        # Create a counter for each possible label (0-9)
        for label in self.legalLabels:
            self.featureCounts[label] = util.Counter()

        # Increments the counts for each feature in the Counter object corresponding to the datum's label
        for features, label in zip(trainingData, trainingLabels):
            self.featureCounts[label] += util.Counter(features)

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
        logJoint = util.Counter(
        )  # Initialize an empty counter to store the log-joint probabilities for each label

        for label in self.legalLabels:  # Iterate over all the possible labels
            priorProb_Labels = math.log(
                self.label_count[label] / self.dataCount)  # Calculate the prior probability of the label using its count in the training data

            featureProb = 0  # store the log-probability of each feature given the label
            for feature, value in datum.items():  # Iterate over all the features in the given datum
                # Calculate the number of times the feature appears in the label with Laplace smoothing
                true_count = self.featureCounts[label][feature] + self.k
                false_count = self.label_count[label] - \
                    self.featureCounts[label][feature] + \
                    self.k  # Calculate the number of times the feature does not appear in the label with Laplace smoothing
                # Calculate the denom of the conditional probability expression
                denom = true_count + false_count

                # Calculate the log-probability of the feature given the label, using Laplace smoothing
                featureProb += math.log(
                    (true_count / denom) if value else (false_count / denom))
            # Calculate the log-joint probability of the label for the given datum
            logJoint[label] = priorProb_Labels + featureProb
        # Return the counter containing the log-joint probabilities for all the labels
        return logJoint
