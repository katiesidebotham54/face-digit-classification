# dataClassifier.py

import naiveBayes
import perceptron
import samples
import sys
import util
import mira
import random
from collections import defaultdict

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features 
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print("===================================")
            print("Mistake on example %d" % i)
            print("Predicted %d; truth is %d" % (prediction, truth))
            print("Image: ")
            print(rawTestData[i])
            break


# =====================
# You don't have to modify any code below.
# =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the 
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form 
        [(2,2), (2, 3), ...] 
        where each tuple represents a pixel.
        """
        image = samples.Datum(None, self.width, self.height)
        for pix in pixels:
            try:
                # This is so that new features that you could define which
                # which are not of the form of (x,y) will not break
                # this image printer...
                x, y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print(image)


def default(str):
    return str + ' [Default: %default]'


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=[
                      'nb', 'naiveBayes', 'perceptron', 'mira'], default='nb')
    parser.add_option('-d', '--data', help=default('Dataset to use'),
                      choices=['digits', 'faces'], default='digits')
    parser.add_option(
        '-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'),
                      default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default(
        "First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'),
                      default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default(
        "Smoothing parameter"), type="float", default=0.1)
    parser.add_option('-i', '--iterations', help=default(
        "Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"),
                      default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    print("training set size:\t" + str(options.training))
    if (options.data == "digits"):
        printImage = ImagePrinter(
            DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        featureFunction = basicFeatureExtractorDigit
    elif (options.data == "faces"):
        printImage = ImagePrinter(
            FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        featureFunction = basicFeatureExtractorFace
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if (options.data == "digits"):
        legalLabels = range(10)
    else:
        legalLabels = range(2)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" %
              options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" %
              options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if (options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        print(options.smoothing)
        classifier.setSmoothing(options.smoothing)
        print("using smoothing parameter k=%f for naivebayes" %
              options.smoothing)
    elif (options.classifier == "perceptron"):
        classifier = perceptron.PerceptronClassifier(
            legalLabels, options.iterations)
    elif (options.classifier == "mira"):
        classifier = mira.MiraClassifier(legalLabels, options.iterations)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options


USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default naiveBayes classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
                 """

# Main harness code


def runClassifier(args, options):

    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data
    numTraining = options.training
    numTest = options.test

    if (options.data == "faces"):
        rawTrainingData = samples.loadDataFile(
            "facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        print("training data: " + str(numTraining))
        trainingLabels = samples.loadLabelsFile(
            "facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile(
            "facedata/facedatatrain", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile(
            "facedata/facedatatrainlabels", numTest)
        rawTestData = samples.loadDataFile(
            "facedata/facedatatest", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile(
            "facedata/facedatatestlabels", numTest)
    else:
        rawTrainingData = samples.loadDataFile(
            "digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile(
            "digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile(
            "digitdata/validationimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile(
            "digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile(
            "digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)

    # Split data and labels
    trainingPairs = list(zip(rawTrainingData, trainingLabels))
    validationPairs = list(zip(rawValidationData, validationLabels))
    testPairs = list(zip(rawTestData, testLabels))

    # Randomize training pairs
    random.shuffle(trainingPairs)
    random.shuffle(validationPairs)
    random.shuffle(testPairs)

    # Unzip shuffled pairs
    shuffledTrainingData, shuffledTrainingLabels = zip(*trainingPairs)
    shuffledValidationData, shuffledValidationLabels = zip(*validationPairs)
    shuffledTestData, shuffledTestLabels = zip(*testPairs)
    # Extract features
    print("Extracting features...")
    trainingData = map(featureFunction, shuffledTrainingData)
    validationData = map(featureFunction, shuffledValidationData)

    # Conduct training and testing
    testData = map(featureFunction, shuffledTestData)
    print("Training...")
    classifier.train(trainingData, shuffledTrainingLabels,
                     validationData, shuffledValidationLabels)
    print("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i]
               for i in range(len(shuffledValidationLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(shuffledValidationLabels)) +
          " (%.1f%%).") % (100.0 * correct / len(shuffledValidationLabels)))
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == shuffledTestLabels[i]
               for i in range(len(testLabels))].count(True)
    print(str("Test_Percent: "), (correct), ("correct out of " + str(len(shuffledTestLabels)) +
          " (%.1f%%).") % (100.0 * correct / len(shuffledTestLabels)))
    analysis(classifier, guesses, testLabels,
             testData, rawTestData, printImage)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args, options)
