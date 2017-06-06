# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from util.activation_functions import Activation
from util.loss_functions import DifferentError
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.01
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100
        #self.bias = np.random.rand()/100
        self.activationFunction = Activation.sign
        self.lossFunction = DifferentError()

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        # Write your code to train the perceptron here
        # TODO: check using epochs right way
        # TODO: check using validation data right way
        if not verbose:
            for _ in xrange(self.epochs):
                output = self.evaluate(self.trainingSet)
                error = list(map(self.lossFunction.calculateError, self.trainingSet.label, output))
                map(lambda x, y: self.updateWeights(x, y), self.trainingSet.input, error)
        else:
            # The same training algorithm with validation and logging
            validationOutput = self.evaluate(self.validationSet)
            logging.info("Validation accuracy %.2f%% before training", \
                accuracy_score(self.validationSet.label, validationOutput) * 100)
            for i in range(self.epochs):
                output = self.evaluate(self.trainingSet)
                error = list(map(self.lossFunction.calculateError, self.trainingSet.label, output))
                map(lambda x, y: self.updateWeights(x, y), self.trainingSet.input, error)
                # Validation
                validationOutput = self.evaluate(self.validationSet)
                logging.info("Validation accuracy %.2f%% at %dth epoch", \
                    accuracy_score(self.validationSet.label, validationOutput) * 100, i)
        # Plot the weight values
        #plt.imshow(self.weight.reshape((28, 28)))
        #plt.colorbar()
        #plt.show()

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        # Perceptron without bias
        # return bool(self.activationFunction(np.dot(self.weight, np.array(testInstance))))
        return bool(self.fire(testInstance))
        
        # Perceptron with bias
        #return bool(self.activationFunction(np.dot(self.weight, np.array(testInstance)) \
        #    + self.bias))

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        self.weight += list(map(lambda x: self.learningRate * error * x, input))
        # Perceptron with bias
        #self.bias += self.learningRate * error
         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # Perceptron without bias
        return self.activationFunction(np.dot(self.weight, np.array(input)))
            
        # Perceptron with bias
        #return self.activationFunction(np.dot(self.weight, np.array(input)) + self.bias)
