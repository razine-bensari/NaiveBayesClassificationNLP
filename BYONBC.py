"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, annes.cherid@gmail.com
"""
from NaiveBayesClassifier import NaiveBayesClassifier
from decimal import Decimal


class BYONBC:
    def __init__(self):
        self.model_ca = NaiveBayesClassifier(0, 3, 0.3, "./training_files/ca_training-tweets.txt", "ca",
                                             18318)
        self.model_gl = NaiveBayesClassifier(0, 3, 0.3, "./training_files/gl_training-tweets.txt", "gl",
                                             18318)
        self.model_en = NaiveBayesClassifier(0, 3, 0.3, "./training_files/en_training-tweets.txt", "en",
                                             18318)
        self.model_es = NaiveBayesClassifier(0, 3, 0.3, "./training_files/es_training-tweets.txt", "es",
                                             18318)
        self.model_pt = NaiveBayesClassifier(0, 3, 0.3, "./training_files/pt_training-tweets.txt", "pt",
                                             18318)
        self.model_eu = NaiveBayesClassifier(0, 3, 0.3, "./training_files/eu_training-tweets.txt", "eu",
                                             18318)
        self.arrayModel = [self.model_ca, self.model_gl, self.model_en, self.model_es, self.model_pt, self.model_eu]
        self.probability = 0
        self.totalTweetCount = 18318

    def trainClassifier(self):
        print("Training BYONBC. . .")
        for model in self.arrayModel:
            model.constructModel()

    def testClassifier(self):
        print("Testing BYONBC. . .")
        trace_file_name = "./trace_files/trace_BYONBC.txt"
        trace_file = open(trace_file_name, "w")
        with open("./OriginalDataSet/test-tweets-given.txt", "r") as file:
            for line in file:
                tweetArray = line.split("\t")
                result = []
                for model in self.arrayModel:
                    result.append((model.calculateProbability(tweetArray[3]), model.language))
                highest_prob = self.get_highest_prob(result)
                if tweetArray[2] == highest_prob[1]:  # Correct
                    prob = "{:.2E}".format(Decimal(highest_prob[0]))
                    s = str(tweetArray[0] + "  " + highest_prob[1] + "  " + str(prob) + "  " + tweetArray[
                        2] + "  " + "correct" "\n")
                    trace_file.write(str(s))
                else:
                    prob = "{:.2E}".format(Decimal(highest_prob[0]))
                    s = str(tweetArray[0] + "  " + highest_prob[1] + "  " + str(prob) + "  " + tweetArray[
                        2] + "  " + "wrong" "\n")
                    trace_file.write(str(s))
        trace_file.close()

    def get_highest_prob(self, result):
        highest_tupl = result[0]
        for tupl in result:
            if tupl[0] > highest_tupl[0]:
                highest_tupl = tupl
        return highest_tupl