"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
import NaiveBayesClassifier as NaiveBayesClassifier

import NaiveBayesClassifierUtils
from BYONBC import BYONBC


# NaiveBayesClassifierUtils.cleanTrainingData()

NaiveBayesClassifierUtils.trainAllNBLanguageClassifier()
NaiveBayesClassifierUtils.generateAllEvalFiles()
NaiveBayesClassifierUtils.generateStatsFile()



# NaiveBayesClassifierUtils.runDemoTest(BYONBC())






NaiveBayesClassifierUtils.trainBYONBC()
NaiveBayesClassifierUtils.generateBYONBCEvalFiles()





# TweetClassifier = NBLanguageClassifier(0, 1, 0.0)
# TweetClassifier.trainClassifier()
# TweetClassifier.testClassifier()








