# mira.py

# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.002
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        # this could be useful for your code later...
        trainingData = list(trainingData)
        self.features = trainingData[0].keys()
        for label in self.legalLabels:
            for feature in self.features:
                self.weights[label][feature] = 0
        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
        then store the weights that give the best accuracy on the validationData.
        Use the provided self.weights[label] data structure so that 
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        # Iterate over each value of C in Cgrid
        for C in Cgrid:
            # Copy the initial weights
            weights = self.weights.copy()
            # Iterate over the training data for a maximum number of iterations
            for iteration in range(self.max_iterations):
                print("Starting iteration ", iteration, "...")
                for i in range(len(trainingData)):
                    # Get the current datum and calculate scores for each possible label
                    datum = trainingData[i]
                    scores = util.Counter()
                    for label in self.legalLabels:
                        scores[label] = weights[label] * datum
                    # Check if the current guess is correct, and if not, update the weights
                    if scores.argMax() != trainingLabels[i]:
                        # Calculate the scaling factor for the update
                        scale_fact = min(C, ((weights[scores.argMax(
                        )] - weights[trainingLabels[i]]) * datum + 1.0) / (2.0 * (datum * datum)))
                        # Calculate the delta to update the weights
                        delta = datum.copy()
                        for feature in delta:
                            delta[feature] *= scale_fact
                        # Update the weights for the correct and incorrect labels
                        weights[trainingLabels[i]] += delta
                        weights[scores.argMax()] -= delta

            # Evaluate the accuracy of the weights on the validation data
            guesses = self.classify(validationData)
            correct = [guesses[i] == validationLabels[i]
                       for i in range(len(validationLabels))]
            accuracy = sum(correct) / len(correct)
            # If the accuracy is better than previous weights, update the weights
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.weights = weights

        # Store the best weights
        self.weights = weights

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
