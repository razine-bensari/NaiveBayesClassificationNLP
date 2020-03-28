"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
from NaiveBayesClassifier import NaiveBayesClassifier

model = NaiveBayesClassifier(2, 1, 0.5, "./training_files/gl_training-tweets.txt", "gl", 18318)

model.constructModel()

prob_base10 = model.calculateProbability("@AnderDelPozo @PesqueWhite hahaha yo tambien me he quedao pillao ahahha à")

print(str(prob_base10) + " for tweet: @AnderDelPozo @PesqueWhite hahaha yo tambien me he quedao pillao ahahha à which is "
                         "es")
