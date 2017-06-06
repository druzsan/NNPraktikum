# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01 * np.random.randn(self.trainingSet.input.shape[1] + 1)
        
        from util.loss_functions import SumSquaredError, DifferentError
        self.loss = SumSquaredError()
        self.lossPrime = DifferentError()

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        # Threshold for convergence condition
        # self.Epsilon = 0.0001
        perfomance = np.array([])
        if not verbose:
            for _ in np.arange(self.epochs):
                output = map(self.fire, self.trainingSet.input)
                error = self.loss.calculateError(self.trainingSet.label, output)
                # Threshold convergence condition
                # if np.abs(error) <= self.Epsilon:
                #     break
                if error == 0.0:
                    break
                errorPrime = self.lossPrime.calculateError(self.trainingSet.label, output)
                grad = np.append(np.matmul(errorPrime, self.trainingSet.input), np.sum(errorPrime))
                self.updateWeights(grad)
        else:
            # The same training algorithm with logging and plotting
            for i in np.arange(self.epochs):
                output = map(self.fire, self.trainingSet.input)
                error = self.loss.calculateError(self.trainingSet.label, output)
                logging.info("Epoch: %i; Error: %f", i, error)
                perfomance = np.append(perfomance, error)
                if error == 0.0:
                    break
                errorPrime = self.lossPrime.calculateError(self.trainingSet.label, output)
                grad = np.append(np.matmul(errorPrime, self.trainingSet.input), np.sum(errorPrime))
                self.updateWeights(grad)
            plt.figure()
            x = np.arange(perfomance.size)
            y = perfomance
            plt.plot(x, y)
            plt.xlabel('epochs')
            plt.ylabel('perfomance (SSE)')
            plt.title('Perfomance of logistic regression with ' + r'$\eta=$' + str(self.learningRate))
            plt.show()
            
        
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
        return self.fire(testInstance) > 0.5

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

    def updateWeights(self, grad):
        # Calculation must work element-wise with arrays as well as with single numbers
        self.weight += self.learningRate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight[:-1]) + self.weight[-1])
