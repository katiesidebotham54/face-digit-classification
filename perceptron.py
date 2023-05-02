# perceptron.py
# -------------

# Perceptron implementation
import util
PRINT = True


class PerceptronClassifier:
    """
     Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        trainingData = list(trainingData)
        self.features = trainingData[0].keys()
        vectors = util.Counter()

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                for label in self.legalLabels:
                    vectors[label] = trainingData[i].__mul__(
                        self.weights[label])
                if not (trainingLabels[i] == vectors.argMax()):
                    self.weights[trainingLabels[i]].__radd__(trainingData[i])
                    self.weights[vectors.argMax()].__sub__(trainingData[i])
          # util.raiseNotDefined()

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
