"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
"""
import numpy as np
import string
import math


def getLanguage(array):
    return array[0]


# BOW is a Bag of Word dictionary using alphabet as keys and frequencies as values
def getFrequencies(BOW, tweet):
    for c in tweet:
        if c in BOW:
            BOW[c] += 1


# BOW is a Bag of Word dictionary using alphabet as keys and frequencies as values
def getFrequenciesForV2(BOW, tweet):
    for c in tweet:
        if c in BOW:
            BOW[c] += 1
        if c not in BOW and c.isalpha():
            BOW[c] = 1
    BOW["<NOT-APPEAR>"] = 0


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


# BOW is a Bag of Word dictionary using alphabet as keys and frequencies as values
def addSmoothing(BOW, delta):
    for c in BOW:
        if BOW[c] == 0:
            BOW[c] += delta


class NaiveBayesClassifier:
    def __init__(self, vocabulary, n, delta, trainingFile, language, totalTweetCount):
        self.typeOfNGrams = None
        self.vocabulary = vocabulary
        self.n = n
        self.delta = delta
        self.trainingFile = trainingFile
        self.array = None
        self.BOW_V0 = dict.fromkeys("abcdefghijklmnopqrstuvwxyz", 0)
        self.BOW_V1 = dict.fromkeys("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 0)
        self.BOW_V2 = dict()
        self.language = language
        self.probability = 0
        self.tweetCount = 0
        self.totalTweetCount = totalTweetCount

    def constructModel(self):
        self.buildNGrams()

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
            self.buildUnigramsWhenVocabularyIsOne()
        if self.vocabulary == 2:
            self.buildUnigramsWhenVocabularyIsTwo()

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

    def buildUnigramsWhenVocabularyIsOne(self):
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                getFrequencies(self.BOW_V1, tweetArray[3])
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print(self.BOW_V1)
        addSmoothing(self.BOW_V1, self.delta)

    def buildUnigramsWhenVocabularyIsTwo(self):
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                getFrequenciesForV2(self.BOW_V2, tweetArray[3])
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print(self.BOW_V2)
        addSmoothing(self.BOW_V2, self.delta)

    def calculateProbability(self, tweet):
        if self.n == 1:
            if self.vocabulary == 0:
                return self.calculate_probability_n1_v0(tweet)
            elif self.vocabulary == 1:
                return self.calculate_probability_n1_v1(tweet)
            elif self.vocabulary == 2:
                return self.calculate_probability_n1_v2(tweet)
            else:
                print("Invalid V")
        elif self.n == 2:
            if self.vocabulary == 0:
                pass
            elif self.vocabulary == 1:
                pass
            elif self.vocabulary == 2:
                pass
            else:
                print("Invalid V")
        elif self.n == 3:
            if self.vocabulary == 0:
                pass
            elif self.vocabulary == 1:
                pass
            elif self.vocabulary == 2:
                pass
            else:
                print("Invalid V")
        else:
            print("Invalid n")

    def calculate_probability_n1_v0(self, tweet):
        sum_of_prob = 0
        total_chars = sum(self.BOW_V0.values())
        print("total char in bow: " + str(total_chars))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prob_of_tweet_base10 = math.log(tweet_ratio, 10)
        for c in tweet:
            if c in self.BOW_V0:
                char_ratio = self.BOW_V0.get(c) / total_chars
                sum_of_prob += math.log(char_ratio, 10)
                print("char probablity: " + str(self.BOW_V0.get(c) / total_chars))
        self.probability = prob_of_tweet_base10 + sum_of_prob
        return self.probability

    def calculate_probability_n1_v1(self, tweet):
        sum_of_prob = 0
        total_chars = sum(self.BOW_V1.values())
        print("total char in bow: " + str(total_chars))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prob_of_tweet_base10 = math.log(tweet_ratio, 10)
        for c in tweet:
            if c in self.BOW_V1:
                char_ratio = self.BOW_V1.get(c) / total_chars
                sum_of_prob += math.log(char_ratio, 10)
                print("char probablity: " + str(self.BOW_V1.get(c) / total_chars))
        self.probability = prob_of_tweet_base10 + sum_of_prob
        return self.probability

    def calculate_probability_n1_v2(self, tweet):
        sum_of_prob = 0
        total_chars = sum(self.BOW_V2.values())
        print("total char in bow: " + str(total_chars))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prob_of_tweet_base10 = math.log(tweet_ratio, 10)
        for c in tweet:
            if c in self.BOW_V2:
                char_ratio = self.BOW_V2.get(c) / total_chars
                sum_of_prob += math.log(char_ratio, 10)
                print("char probablity: " + str(self.BOW_V2.get(c) / total_chars))
            if c not in self.BOW_V2 and c.isalpha():
                print("This character was not in training: " + str(c))
                #TODO get more info on what to do here
        self.probability = prob_of_tweet_base10 + sum_of_prob
        return self.probability