"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
from NaiveBayesClassifier import NaiveBayesClassifier

# model_ca = NaiveBayesClassifier(1, 2, 0.5, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_gl = NaiveBayesClassifier(1, 2, 0.5, "./training_files/gl_training-tweets.txt", "gl", 18318)
# model_en = NaiveBayesClassifier(1, 2, 0.5, "./training_files/en_training-tweets.txt", "en", 18318)
# model_es = NaiveBayesClassifier(1, 2, 0.5, "./training_files/es_training-tweets.txt", "es", 18318)
# model_pt = NaiveBayesClassifier(1, 2, 0.5, "./training_files/pt_training-tweets.txt", "pt", 18318)
# model_eu = NaiveBayesClassifier(1, 2, 0.5, "./training_files/eu_training-tweets.txt", "eu", 18318)
#
# model_ca.constructModel()
# model_gl.constructModel()
# model_en.constructModel()
# model_es.constructModel()
# model_pt.constructModel()
# model_eu.constructModel()
#
# prob_base10_ca = model_ca.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_gl = model_gl.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_en = model_en.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_es = model_es.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_pt = model_pt.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_eu = model_eu.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# print("Tweet is ca.")

# prob_base10_ca = model_ca.calculateProbability("@esport3 @marca no tenemos bastante con Madrid, que al Barça le crecen los enanos, un socio desestabiliza su entidad #incomprensible")
# prob_base10_gl = model_gl.calculateProbability("@esport3 @marca no tenemos bastante con Madrid, que al Barça le crecen los enanos, un socio desestabiliza su entidad #incomprensible")
# prob_base10_en = model_en.calculateProbability("@esport3 @marca no tenemos bastante con Madrid, que al Barça le crecen los enanos, un socio desestabiliza su entidad #incomprensible")
# prob_base10_es = model_es.calculateProbability("@esport3 @marca no tenemos bastante con Madrid, que al Barça le crecen los enanos, un socio desestabiliza su entidad #incomprensible")
# prob_base10_pt = model_pt.calculateProbability("@esport3 @marca no tenemos bastante con Madrid, que al Barça le crecen los enanos, un socio desestabiliza su entidad #incomprensible")
# prob_base10_eu = model_eu.calculateProbability("@esport3 @marca no tenemos bastante con Madrid, que al Barça le crecen los enanos, un socio desestabiliza su entidad #incomprensible")
#print("Tweet is es.")

# print("model_ca: " + str(prob_base10_ca))
# print("model_gl: " + str(prob_base10_gl))
# print("model_en: " + str(prob_base10_en))
# print("model_es: " + str(prob_base10_es))
# print("model_pt: " + str(prob_base10_pt))
# print("model_pt: " + str(prob_base10_eu))

# model_ca = NaiveBayesClassifier(2, 2, 0, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_ca.constructModel()
# prob_base10_ca = model_ca.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# print("Tweet is ca.")
# print("model_ca: " + str(prob_base10_ca))

# model_ca = NaiveBayesClassifier(2, 2, 0.5, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_gl = NaiveBayesClassifier(2, 2, 0.5, "./training_files/gl_training-tweets.txt", "gl", 18318)
# model_en = NaiveBayesClassifier(2, 2, 0.5, "./training_files/en_training-tweets.txt", "en", 18318)
# model_es = NaiveBayesClassifier(2, 2, 0.5, "./training_files/es_training-tweets.txt", "es", 18318)
# model_pt = NaiveBayesClassifier(2, 2, 0.5, "./training_files/pt_training-tweets.txt", "pt", 18318)
# model_eu = NaiveBayesClassifier(2, 2, 0.5, "./training_files/eu_training-tweets.txt", "eu", 18318)
#
# model_ca.constructModel()
# model_gl.constructModel()
# model_en.constructModel()
# model_es.constructModel()
# model_pt.constructModel()
# model_eu.constructModel()
#
# prob_base10_ca = model_ca.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_gl = model_gl.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_en = model_en.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_es = model_es.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_pt = model_pt.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# prob_base10_eu = model_eu.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# print("Tweet is ca.")
#
# print("model_ca: " + str(prob_base10_ca))
# print("model_gl: " + str(prob_base10_gl))
# print("model_en: " + str(prob_base10_en))
# print("model_es: " + str(prob_base10_es))
# print("model_pt: " + str(prob_base10_pt))
# print("model_pt: " + str(prob_base10_eu))