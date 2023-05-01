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
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
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
        Outside shell to call your method
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

        # Count the features for each label in the training set
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
        logJoint = util.Counter()

        for label in self.legalLabels:
            priorProb_Labels = math.log(
                self.label_count[label] / self.dataCount)

            featureProb_givenLabel = 0
            for feature, value in datum.items():
                true_count = self.featureCounts[label][feature] + self.k
                false_count = self.label_count[label] - \
                    self.featureCounts[label][feature] + self.k
                denominator = true_count + false_count

                featureProb_givenLabel += math.log(
                    (true_count / denominator) if value else (false_count / denominator))
            logJoint[label] = priorProb_Labels + featureProb_givenLabel
        return logJoint
