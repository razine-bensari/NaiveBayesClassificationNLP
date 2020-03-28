"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
from NaiveBayesClassifier import NaiveBayesClassifier

model_ca = NaiveBayesClassifier(0, 1, 0.5, "./training_files/ca_training-tweets.txt", "ca", 18318)
model_gl = NaiveBayesClassifier(0, 1, 0.5, "./training_files/gl_training-tweets.txt", "gl", 18318)
model_en = NaiveBayesClassifier(0, 1, 0.5, "./training_files/en_training-tweets.txt", "en", 18318)
model_es = NaiveBayesClassifier(0, 1, 0.5, "./training_files/es_training-tweets.txt", "es", 18318)
model_pt = NaiveBayesClassifier(0, 1, 0.5, "./training_files/pt_training-tweets.txt", "pt", 18318)
model_eu = NaiveBayesClassifier(0, 1, 0.5, "./training_files/eu_training-tweets.txt", "eu", 18318)

model_ca.constructModel()
model_gl.constructModel()
model_en.constructModel()
model_es.constructModel()
model_pt.constructModel()
model_eu.constructModel()

prob_base10_ca = model_ca.calculateProbability("Bona victòria; acabant jugant contra 3 i assegurant el passe a la fase final per l'ascens de categoria. El millor de tot és que seguim vius.")
prob_base10_gl = model_gl.calculateProbability("Bona victòria; acabant jugant contra 3 i assegurant el passe a la fase final per l'ascens de categoria. El millor de tot és que seguim vius.")
prob_base10_en = model_en.calculateProbability("Bona victòria; acabant jugant contra 3 i assegurant el passe a la fase final per l'ascens de categoria. El millor de tot és que seguim vius.")
prob_base10_es = model_es.calculateProbability("Bona victòria; acabant jugant contra 3 i assegurant el passe a la fase final per l'ascens de categoria. El millor de tot és que seguim vius.")
prob_base10_pt = model_pt.calculateProbability("Bona victòria; acabant jugant contra 3 i assegurant el passe a la fase final per l'ascens de categoria. El millor de tot és que seguim vius.")
prob_base10_eu = model_eu.calculateProbability("Bona victòria; acabant jugant contra 3 i assegurant el passe a la fase final per l'ascens de categoria. El millor de tot és que seguim vius.")

print("Tweet is es.")
print("model_ca: " + str(prob_base10_ca))
print("model_gl: " + str(prob_base10_gl))
print("model_en: " + str(prob_base10_en))
print("model_es: " + str(prob_base10_es))
print("model_pt: " + str(prob_base10_pt))
print("model_pt: " + str(prob_base10_eu))
