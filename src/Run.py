#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

import numpy
def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=False)                                    
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           learningRate=0.005,
                                           epochs=30)                                  
    
    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")
    print("\nMLP has been training..")
    myMLPClassifier.train()
    print("Done..")
    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Multy-Layer Perceptron recognizer:")
    #evaluator.printComparison(data.testSet, mlpPred)    
    evaluator.printAccuracy(data.testSet, mlpPred)
    
    # Draw
    plot = PerformancePlot("Logistic Regression validation")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)
    
if __name__ == '__main__':
    main()
