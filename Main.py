"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
import NaiveBayesClassifierUtils

# NaiveBayesClassifierUtils.cleanTrainingData()
from NBLanguageClassifier import NBLanguageClassifier

# NaiveBayesClassifierUtils.trainAllNBLanguageClassifier()
# NaiveBayesClassifierUtils.generateAllEvalFiles()
# NaiveBayesClassifierUtils.generateStatsFile()

NaiveBayesClassifierUtils.trainBYONBC()
NaiveBayesClassifierUtils.generateBYONBCEvalFiles()

# TweetClassifier = NBLanguageClassifier(0, 2, 0.0)
# TweetClassifier.trainClassifier()
# TweetClassifier.testClassifier()




