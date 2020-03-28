"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
from NaiveBayesClassifier import NaiveBayesClassifier

model = NaiveBayesClassifier(0, 1, 0, "./training_files/gl_training-tweets.txt", "empty.txt")

model.constructModel()
