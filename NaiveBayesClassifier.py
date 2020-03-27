"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, annes.cherid@gmail.com
"""
import numpy as np


class NaiveBayesClassifier:
    def __init__(self, vocabulary, n, delta, trainingFile, testingFile):
        self.vocabulary = vocabulary
        self.n = n
        self.delta = delta
        self.trainingFile = trainingFile
        self.testingFile = testingFile
        self.array = None
