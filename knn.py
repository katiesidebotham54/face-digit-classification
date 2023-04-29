# knn.py
import util
import numpy
import math
import statistics
from collections import defaultdict


class knnClassifier:
  """
  k-Nearest Neighbor classifier.
  """
  def __init__( self, legalLabels, k=10):
    self.legalLabels = legalLabels
    self.type = "kNN"
    self.k = k
    self.weights = {}

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    Preprocess the training data and save it for later use.
    """
    trainingData = list(trainingData)
    self.trainingData = trainingData
    self.trainingLabels = trainingLabels

  def findDistance(self, test_datum, train_datum):
    """
    Calculate the Euclidean distance between two data points.
    """
    x = test_datum - train_datum
    # return numpy.sum(numpy.abs([x[value] for value in x]))
    return numpy.sqrt(numpy.sum(numpy.square(x)))
    
  def classify(self, data ):
    """
    Find the k closest 'neighbors' of the test image in the training data
    and then return the label which appeared the most. If there is a tie
    then pick the label of the training image with the lowest distance.
    """
    guesses = []
    for datum in data:
        distanceValues = []
        for i in range(len(self.trainingData)):
            distanceValues.append((self.findDistance(datum,self.trainingData[i]), i))
        distanceValues.sort()
        distanceValues = distanceValues[:self.k]
        bestK_labels = []
        for distance in distanceValues:
            bestK_labels.append(self.trainingLabels[distance[1]])
        try:
            guesses.append(statistics.mode(bestK_labels))
        except:
            guesses.append(bestK_labels[0])

    return guesses


