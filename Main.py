"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, cherid.annes@gmail.com
"""
from NBLanguageClassifier import NBLanguageClassifier
import time

# model_ca = NaiveBayesClassifier(2, 3, 0.5, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_gl = NaiveBayesClassifier(2, 3, 0.5, "./training_files/gl_training-tweets.txt", "gl", 18318)
# model_en = NaiveBayesClassifier(2, 3, 0.5, "./training_files/en_training-tweets.txt", "en", 18318)
# model_es = NaiveBayesClassifier(2, 3, 0.5, "./training_files/es_training-tweets.txt", "es", 18318)
# model_pt = NaiveBayesClassifier(2, 3, 0.5, "./training_files/pt_training-tweets.txt", "pt", 18318)
# model_eu = NaiveBayesClassifier(2, 3, 0.5, "./training_files/eu_training-tweets.txt", "eu", 18318)
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
# print("Tweet is es.")

# print("model_ca: " + str(prob_base10_ca))
# print("model_gl: " + str(prob_base10_gl))
# print("model_en: " + str(prob_base10_en))
# print("model_es: " + str(prob_base10_es))
# print("model_pt: " + str(prob_base10_pt))
# print("model_eu: " + str(prob_base10_eu))

# model_ca = NaiveBayesClassifier(1, 3, 0.5, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_ca.constructModel()
# prob_base10_ca = model_ca.calculateProbability("He tingut l'honor d Presidir l'acte de presentació del llibre de fotos cedides x la familia Solà-Llirinos acte molt emotiu")
# print("Tweet is ca.")
# print("model_ca: " + str(prob_base10_ca))

# model_ca = NaiveBayesClassifier(0, 3, 0.5, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_gl = NaiveBayesClassifier(0, 3, 0.5, "./training_files/gl_training-tweets.txt", "gl", 18318)
# model_en = NaiveBayesClassifier(0, 3, 0.5, "./training_files/en_training-tweets.txt", "en", 18318)
# model_es = NaiveBayesClassifier(0, 3, 0.5, "./training_files/es_training-tweets.txt", "es", 18318)
# model_pt = NaiveBayesClassifier(0, 3, 0.5, "./training_files/pt_training-tweets.txt", "pt", 18318)
# model_eu = NaiveBayesClassifier(0, 3, 0.5, "./training_files/eu_training-tweets.txt", "eu", 18318)
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
# print("model_eu: " + str(prob_base10_eu))

# model_ca = NaiveBayesClassifier(1, 3, 1, "./training_files/ca_training-tweets.txt", "ca", 18318)
# model_gl = NaiveBayesClassifier(1, 3, 1, "./training_files/gl_training-tweets.txt", "gl", 18318)
# model_en = NaiveBayesClassifier(1, 3, 1, "./training_files/en_training-tweets.txt", "en", 18318)
# model_es = NaiveBayesClassifier(1, 3, 1, "./training_files/es_training-tweets.txt", "es", 18318)
# model_pt = NaiveBayesClassifier(1, 3, 1, "./training_files/pt_training-tweets.txt", "pt", 18318)
# model_eu = NaiveBayesClassifier(1, 3, 1, "./training_files/eu_training-tweets.txt", "eu", 18318)
#
# model_ca.constructModel()
# model_gl.constructModel()
# model_en.constructModel()
# model_es.constructModel()
# model_pt.constructModel()
# model_eu.constructModel()
#
# prob_base10_ca = model_ca.calculateProbability("Parem de discutir e de se espigar em pleno Twitter isso causa mau ambiente")
# prob_base10_gl = model_gl.calculateProbability("Parem de discutir e de se espigar em pleno Twitter isso causa mau ambiente")
# prob_base10_en = model_en.calculateProbability("Parem de discutir e de se espigar em pleno Twitter isso causa mau ambiente")
# prob_base10_es = model_es.calculateProbability("Parem de discutir e de se espigar em pleno Twitter isso causa mau ambiente")
# prob_base10_pt = model_pt.calculateProbability("Parem de discutir e de se espigar em pleno Twitter isso causa mau ambiente")
# prob_base10_eu = model_eu.calculateProbability("Parem de discutir e de se espigar em pleno Twitter isso causa mau ambiente")
# print("Tweet is pt.")
#
# print("model_ca: " + str(prob_base10_ca))
# print("model_gl: " + str(prob_base10_gl))
# print("model_en: " + str(prob_base10_en))
# print("model_es: " + str(prob_base10_es))
# print("model_pt: " + str(prob_base10_pt))
# print("model_eu: " + str(prob_base10_eu))


startP = time.time()
for v in range(3):
    startV = time.time()
    for n in range(1, 4):
        startN = time.time()
        for d in range(1, 10):
            startD = time.time()
            delta = d * 0.1
            TweetClassifier = NBLanguageClassifier(v, n, round(delta, 1))
            TweetClassifier.trainClassifier()
            TweetClassifier.testClassifier()
            stopD = time.time()
            d = round(delta, 1)
            print("\t\t\tIn V = " + str(v) + ", N = " + str(n) + ", D = " + str(d) + ". Time (s) to iterate over D is " + str(round(stopD - startD, 1)))
        stopN = time.time()
        print("\t\tTime (s) to iterate over N = " + str(n) + ", " + str(round(stopN - startN, 1)))
    stopV = time.time()
    print("\tTime (s) to iterate over V = " + str(v) + ", " + str(round(stopV - startV, 1)))
stopP = time.time()
print("Time for the whole process: " + str(round(stopP - startP, 1)))
