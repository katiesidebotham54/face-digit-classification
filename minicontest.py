# minicontest.py

import util
import classificationMethod


class contestClassifier(classificationMethod.ClassificationMethod):
    """
    Create any sort of classifier you want. You might copy over one of your
    existing classifiers and improve it.
    """

    def __init__(self, legalLabels):
        self.guess = None
        self.type = "minicontest"

    def train(self, data, labels, validationData, validationLabels):
        """
        Please describe your training procedure here.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classify(self, testData):
        """
        Please describe how data is classified here.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
