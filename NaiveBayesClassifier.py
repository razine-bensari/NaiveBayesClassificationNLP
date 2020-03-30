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


def buildTrigramsWhenVocabularyIsTwo():
    None


# BOW is a Bag of Word dictionary using alphabet as keys and frequencies as values
def addSmoothing(BOW, delta):
    for c in BOW:
        if BOW[c] == 0:
            BOW[c] += delta


def getBigrams(BOW_V0, tweet):
    bigrams = []
    i = 0
    max_length = len(tweet)
    while i < max_length:
        if i + 1 >= max_length:
            break
        elif tweet[i] in BOW_V0 and tweet[i + 1] in BOW_V0:
            bigrams.append([tweet[i], tweet[i + 1]])
        i += 1
    return bigrams


def getTrigrams(BOW_V0, tweet):
    trigrams = []
    i = 0
    max_length = len(tweet)
    while i < max_length:
        if i + 2 >= max_length:
            break
        elif tweet[i] in BOW_V0 and tweet[i + 1] in BOW_V0 and tweet[i + 2] in BOW_V0:
            trigrams.append([tweet[i], tweet[i + 1], tweet[i + 2]])
        i += 1
    return trigrams


def addSmoothingBigrams(array, delta):
    for x in np.nditer(array, op_flags=['readwrite']):
        x[...] = x + delta


def addSmoothingTrigrams(array, delta):
    for x in np.nditer(array, op_flags=['readwrite']):
        x[...] = x + delta


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
        self.map_char_to_index = None
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
            self.buildBigramsWhenVocabularyIsZero()
        if self.vocabulary == 1:
            self.buildBigramsWhenVocabularyIsOne()
        if self.vocabulary == 2:
            self.buildBigramsWhenVocabularyIsTwo()

    def buildTrigrams(self):
        if self.vocabulary == 0:
            self.buildTrigramsWhenVocabularyIsZero()
        if self.vocabulary == 1:
            self.buildTrigramsWhenVocabularyIsOne()
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
                return self.calculate_probability_n2_v0(tweet)
            elif self.vocabulary == 1:
                return self.calculate_probability_n2_v1(tweet)
            elif self.vocabulary == 2:
                return self.calculate_probability_n2_v2(tweet)
            else:
                print("Invalid V")
        elif self.n == 3:
            if self.vocabulary == 0:
                return self.calculate_probability_n3_v0(tweet)
            elif self.vocabulary == 1:
                return self.calculate_probability_n3_v1(tweet)
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
                print("char probablity: " + str(char_ratio))
        self.probability = prob_of_tweet_base10 + sum_of_prob
        return self.probability

    def calculate_probability_n1_v1(self, tweet):
        sum_of_prob = 0
        total_chars = sum(self.BOW_V1.values())
        print("total char in bow: " + str(total_chars))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        for c in tweet:
            if c in self.BOW_V1:
                char_ratio = self.BOW_V1.get(c) / total_chars
                sum_of_prob += math.log(char_ratio, 10)
                print("char probablity: " + str(char_ratio))
        self.probability = prior_of_tweet_base10 + sum_of_prob
        self.probability = sum_of_prob
        return self.probability

    def calculate_probability_n1_v2(self, tweet):
        sum_of_prob = 0
        total_chars = sum(self.BOW_V2.values())
        print("total char in bow: " + str(total_chars))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        for c in tweet:
            if c in self.BOW_V2:
                char_ratio = self.BOW_V2.get(c) / total_chars
                sum_of_prob += math.log(char_ratio, 10)
                print("char probablity: " + str(char_ratio))
            if c not in self.BOW_V2 and c.isalpha():
                print("This character was not in training: " + str(c))
                # TODO get more info on what to do here
        self.probability = prior_of_tweet_base10 + sum_of_prob
        return self.probability

    def buildBigramsWhenVocabularyIsZero(self):
        self.array = np.zeros([27, 27])  # extra row and column for <NOT-APPEAR>
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                bigrams_couple = getBigrams(self.BOW_V0, tweetArray[3])
                if len(bigrams_couple):
                    self.populateBigram_V0(bigrams_couple, self.array)
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print("This is the vocabulary: ")
        print(self.BOW_V0)
        addSmoothingBigrams(self.array, self.delta)

    def buildBigramsWhenVocabularyIsOne(self):
        self.array = np.zeros([53, 53])  # extra row and column for <NOT-APPEAR>
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                bigrams_couple = getBigrams(self.BOW_V1, tweetArray[3])
                if len(bigrams_couple):
                    self.populateBigram_V1(bigrams_couple, self.array)
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print("This is the vocabulary: ")
        print(self.BOW_V1)
        addSmoothingBigrams(self.array, self.delta)

    def buildBigramsWhenVocabularyIsTwo(self):
        self.map_char_to_index = self.get_map_vocabular(self.trainingFile)
        self.array = np.zeros(
            [len(self.map_char_to_index), len(self.map_char_to_index)])  # Not appear column is already there
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                bigrams_couple = getBigrams(self.map_char_to_index, tweetArray[3])
                if len(bigrams_couple):
                    self.populateBigram_V2(bigrams_couple, self.array, self.map_char_to_index)
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print("This is the vocabulary: ")
        print(self.map_char_to_index)
        addSmoothingBigrams(self.array, self.delta)

    def get_map_vocabular(self, trainingFile):
        map_char_to_index = dict()
        index = 0
        with open(trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                for c in tweetArray[3]:
                    if c.isalpha() and c not in map_char_to_index:
                        map_char_to_index[c] = index
                        index += 1
        map_char_to_index["<NOT-APPEAR>"] = 0.0
        return map_char_to_index

    def populateBigram_V0(self, bigrams_couple, array):
        for bigram in bigrams_couple:
            x = string.ascii_lowercase.index(bigram[0])
            y = string.ascii_lowercase.index(bigram[1])
            array[x, y] += 1

    def populateBigram_V1(self, bigrams_couple, array):
        for bigram in bigrams_couple:
            if bigram[0].islower() and bigram[1].islower():
                x = string.ascii_lowercase.index(bigram[0])
                y = string.ascii_lowercase.index(bigram[1])
                array[x, y] += 1
            elif bigram[0].islower() and bigram[1].isupper():
                x = string.ascii_lowercase.index(bigram[0])
                y = string.ascii_uppercase.index(bigram[1])
                array[x, y + 26] += 1
            elif bigram[0].isupper() and bigram[1].islower():
                x = string.ascii_uppercase.index(bigram[0])
                y = string.ascii_lowercase.index(bigram[1])
                array[x + 26, y] += 1
            elif bigram[0].isupper() and bigram[1].isupper():
                x = string.ascii_uppercase.index(bigram[0])
                y = string.ascii_uppercase.index(bigram[1])
                array[x + 26, y + 26] += 1
            else:
                print("Not able to populate bigram")
                print(bigram)

    def populateBigram_V2(self, bigrams_couple, array, map_char_to_index):
        for bigram in bigrams_couple:
            x = int(map_char_to_index[bigram[0]])
            y = int(map_char_to_index[bigram[1]])
            array[x, y] += 1

    def calculate_probability_n2_v0(self, tweet):
        sum_of_prob = 0
        total_chars_row = self.array.sum(axis=1)
        print("total char in each row: " + str(total_chars_row))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        bigrams_couple = getBigrams(self.BOW_V0, tweet)
        for bigram in bigrams_couple:
            x = string.ascii_lowercase.index(bigram[0])
            y = string.ascii_lowercase.index(bigram[1])
            bigram_ratio = self.array[x, y] / total_chars_row[x]
            sum_of_prob += math.log(bigram_ratio, 10)
            print("bigram probablity: " + str(bigram_ratio))
        self.probability = prior_of_tweet_base10 + sum_of_prob
        return self.probability

    def calculate_probability_n2_v1(self, tweet):
        sum_of_prob = 0
        total_chars_row = self.array.sum(axis=1)
        print("total char in each row: " + str(total_chars_row))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        bigrams_couple = getBigrams(self.BOW_V1, tweet)
        for bigram in bigrams_couple:
            if bigram[0].islower() and bigram[1].islower():
                x = string.ascii_lowercase.index(bigram[0])
                y = string.ascii_lowercase.index(bigram[1])
                bigram_ratio = self.array[x, y] / total_chars_row[x]
                sum_of_prob += math.log(bigram_ratio, 10)
                print("bigram probablity: " + str(bigram_ratio))
            elif bigram[0].islower() and bigram[1].isupper():
                x = string.ascii_lowercase.index(bigram[0])
                y = string.ascii_uppercase.index(bigram[1])
                bigram_ratio = self.array[x, y + 26] / total_chars_row[x]
                sum_of_prob += math.log(bigram_ratio, 10)
                print("bigram probablity: " + str(bigram_ratio))
            elif bigram[0].isupper() and bigram[1].islower():
                x = string.ascii_uppercase.index(bigram[0])
                y = string.ascii_lowercase.index(bigram[1])
                bigram_ratio = self.array[x + 26, y] / total_chars_row[x + 26]
                sum_of_prob += math.log(bigram_ratio, 10)
                print("bigram probablity: " + str(bigram_ratio))
            elif bigram[0].isupper() and bigram[1].isupper():
                x = string.ascii_uppercase.index(bigram[0])
                y = string.ascii_uppercase.index(bigram[1])
                bigram_ratio = self.array[x + 26, y + 26] / total_chars_row[x + 26]
                sum_of_prob += math.log(bigram_ratio, 10)
                print("bigram probablity: " + str(bigram_ratio))
            else:
                print("Not able to calculate probability")
                print(bigram)
                return 0.0
        self.probability = prior_of_tweet_base10 + sum_of_prob
        return self.probability

    def calculate_probability_n2_v2(self, tweet):
        sum_of_prob = 0
        total_chars_row = self.array.sum(axis=1)
        print("total char in each row: " + str(total_chars_row))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        bigrams_couple = getBigrams(self.BOW_V0, tweet)
        for bigram in bigrams_couple:
            x = self.map_char_to_index[bigram[0]]
            y = self.map_char_to_index[bigram[1]]
            bigram_ratio = self.array[x, y] / total_chars_row[x]
            sum_of_prob += math.log(bigram_ratio, 10)
            print("bigram probablity: " + str(bigram_ratio))
        self.probability = prior_of_tweet_base10 + sum_of_prob
        return self.probability

    def buildTrigramsWhenVocabularyIsZero(self):
        self.array = np.zeros([27, 27, 27])
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                trigram_couple = getTrigrams(self.BOW_V0, tweetArray[3])
                if len(trigram_couple):
                    self.populateTrigram_V0(trigram_couple, self.array)
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print("This is the vocabulary: ")
        print(self.BOW_V0)
        addSmoothingTrigrams(self.array, self.delta)

    def buildTrigramsWhenVocabularyIsOne(self):
        self.array = np.zeros([53, 53, 53])
        self.language = getLanguage(self.trainingFile.split("_"))
        tweetCount = 0
        with open(self.trainingFile, "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                trigram_couple = getTrigrams(self.BOW_V1, tweetArray[3])
                if len(trigram_couple):
                    self.populateTrigram_V1(trigram_couple, self.array)
                tweetCount += 1
        self.tweetCount = tweetCount
        print("Tweet count: " + str(tweetCount))
        print("This is the vocabulary: ")
        print(self.BOW_V1)
        addSmoothingTrigrams(self.array, self.delta)

    def populateTrigram_V0(self, trigram_couple, array):
        for trigram in trigram_couple:
            x = string.ascii_lowercase.index(trigram[0])
            y = string.ascii_lowercase.index(trigram[1])
            z = string.ascii_lowercase.index(trigram[2])
            array[x, y, z] += 1

    def populateTrigram_V1(self, trigram_couple, array):
        for trigram in trigram_couple:
            if trigram[0].islower() and trigram[1].islower() and trigram[2].islower():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                array[x, y, z] += 1
            elif trigram[0].islower() and trigram[1].isupper() and trigram[2].islower():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                array[x, y + 26, z] += 1
            elif trigram[0].isupper() and trigram[1].islower() and trigram[2].islower():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                array[x + 26, y, z] += 1
            elif trigram[0].isupper() and trigram[1].isupper() and trigram[2].islower():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                array[x + 26, y + 26, z] += 1
            elif trigram[0].islower() and trigram[1].islower() and trigram[2].isupper():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                array[x, y, z + 26] += 1
            elif trigram[0].islower() and trigram[1].isupper() and trigram[2].isupper():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                array[x, y + 26, z + 26] += 1
            elif trigram[0].isupper() and trigram[1].islower() and trigram[2].isupper():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                array[x + 26, y, z + 26] += 1
            elif trigram[0].isupper() and trigram[1].isupper() and trigram[2].isupper():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                array[x + 26, y + 26, z + 26] += 1
            else:
                print("Not able to populate trigram")
                print(trigram)

    def calculate_probability_n3_v0(self, tweet):
        sum_of_prob = 0
        total_chars_matrix = self.array.sum(axis=2)
        total_chars_row = total_chars_matrix.sum(axis=1)
        print("total char in each row: " + str(total_chars_row))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        trigrams_couple = getTrigrams(self.BOW_V0, tweet)
        for trigram in trigrams_couple:
            x = string.ascii_lowercase.index(trigram[0])
            y = string.ascii_lowercase.index(trigram[1])
            z = string.ascii_lowercase.index(trigram[2])
            print(x, y, z)
            trigram_ratio = self.array[x, y, z] / total_chars_row[x]
            sum_of_prob += math.log(trigram_ratio, 10)
            print("bigram probablity: " + str(trigram_ratio))
        self.probability = prior_of_tweet_base10 + sum_of_prob
        return self.probability

    def calculate_probability_n3_v1(self, tweet):
        sum_of_prob = 0
        total_chars_matrix = self.array.sum(axis=2)
        total_chars_row = total_chars_matrix.sum(axis=1)
        print("total char in each row: " + str(total_chars_row))
        tweet_ratio = self.tweetCount / self.totalTweetCount
        print("Tweet probablity: " + str(tweet_ratio))
        prior_of_tweet_base10 = math.log(tweet_ratio, 10)
        trigrams_couple = getTrigrams(self.BOW_V1, tweet)
        for trigram in trigrams_couple:
            if trigram[0].islower() and trigram[1].islower() and trigram[2].islower():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                trigram_ratio = self.array[x, y, z] / total_chars_row[x]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].islower() and trigram[1].isupper() and trigram[2].islower():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                trigram_ratio = self.array[x, y + 26, z] / total_chars_row[x]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].isupper() and trigram[1].islower() and trigram[2].islower():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                trigram_ratio = self.array[x + 26, y, z] / total_chars_row[x + 26]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].isupper() and trigram[1].isupper() and trigram[2].islower():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_lowercase.index(trigram[2])
                trigram_ratio = self.array[x + 26, y + 26, z] / total_chars_row[x + 26]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].islower() and trigram[1].islower() and trigram[2].isupper():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                trigram_ratio = self.array[x, y, z + 26] / total_chars_row[x]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].islower() and trigram[1].isupper() and trigram[2].isupper():
                x = string.ascii_lowercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                trigram_ratio = self.array[x, y + 26, z + 26] / total_chars_row[x]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].isupper() and trigram[1].islower() and trigram[2].isupper():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_lowercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                trigram_ratio = self.array[x + 26, y, z + 26] / total_chars_row[x + 26]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            elif trigram[0].isupper() and trigram[1].isupper() and trigram[2].isupper():
                x = string.ascii_uppercase.index(trigram[0])
                y = string.ascii_uppercase.index(trigram[1])
                z = string.ascii_uppercase.index(trigram[2])
                trigram_ratio = self.array[x + 26, y + 26, z + 26] / total_chars_row[x + 26]
                sum_of_prob += math.log(trigram_ratio, 10)
                print("bigram probablity: " + str(trigram_ratio))
            else:
                print("Not able to calculate probability")
                print(trigram)
                return 0.0
        self.probability = prior_of_tweet_base10 + sum_of_prob
        return self.probability
