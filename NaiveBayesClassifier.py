"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
"""
import numpy as np
import string


def getLanguage(array):
    return array[0]


def getFrequencies(UV0, tweet):
    for c in tweet:
        if c in UV0:
            UV0[c] += 1


def buildUnigramsWhenVocabularyIsOne():
    None


def buildUnigramsWhenVocabularyIsTwo():
    None


def buildBigramsWhenVocabularyIsZero():
    None


def buildBigramsWhenVocabularyIsOne():
    None


def buildBigramsWhenVocabularyIsTwo():
    None


def buildTrigramsWhenVocabularyIsZero():
    None


def buildTrigramsWhenVocabularyIsOne():
    None


def buildTrigramsWhenVocabularyIsTwo():
    None


def addSmoothing(BOW_V0, delta):
    for c in BOW_V0:
        if BOW_V0[c] == 0:
            BOW_V0[c] += delta


class NaiveBayesClassifier:
    def __init__(self, vocabulary, n, delta, trainingFile, testingFile):
        self.typeOfNGrams = None
        self.vocabulary = vocabulary
        self.n = n
        self.delta = delta
        self.trainingFile = trainingFile
        self.testingFile = testingFile
        self.array = None
        self.BOW_V0 = dict.fromkeys("abcdefghijklmnopqrstuvwxyz", 0)
        self.BOW_V1 = dict.fromkeys("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 0)
        self.BOW_V2 = dict()
        self.language = ""
        self.probability = 0
        self.tweetCount = 0

    def constructModel(self):
        self.buildNGrams()
        # self.testModel()

    def buildNGrams(self):
        if self.n == 1:
            self.typeOfNGrams = "Unigrams"
            self.buildUnigrams()
        if self.n == 2:
            self.typeOfNGrams = "Bigrams"
            self.buildBigrams()
        if self.n == 3:
            self.typeOfNGrams = "Trigrams"
            self.buildTrigrams()

    def buildUnigrams(self):
        if self.vocabulary == 0:
            self.buildUnigramsWhenVocabularyIsZero()
        if self.vocabulary == 1:
            buildUnigramsWhenVocabularyIsOne()
        if self.vocabulary == 2:
            buildUnigramsWhenVocabularyIsTwo()

    def buildBigrams(self):
        if self.vocabulary == 0:
            buildBigramsWhenVocabularyIsZero()
        if self.vocabulary == 1:
            buildBigramsWhenVocabularyIsOne()
        if self.vocabulary == 2:
            buildBigramsWhenVocabularyIsTwo()

    def buildTrigrams(self):
        if self.vocabulary == 0:
            buildTrigramsWhenVocabularyIsZero()
        if self.vocabulary == 1:
            buildTrigramsWhenVocabularyIsOne()
        if self.vocabulary == 2:
            buildTrigramsWhenVocabularyIsTwo()

    def buildUnigramsWhenVocabularyIsZero(self):
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                getFrequencies(self.BOW_V0, tweetArray[3])
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print(self.BOW_V0)
        addSmoothing(self.BOW_V0, self.delta)
