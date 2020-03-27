"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, annes.cherid@gmail.com

Class with helper function for Naive Bayes classifers.
Helps clean the data and provide insight on the classifier and produce output files required
"""
from enum import Enum


class Language(Enum):
    EU = "eu"
    CA = "ca"
    GL = "gl"
    ES = "es"
    EN = "en"
    PT = "pt"


print("Creating training files for each classes")
fileName_EU = "./training_files/" + Language.EU.value + "_training-tweets.txt"
fileName_CA = "./training_files/" + Language.CA.value + "_training-tweets.txt"
fileName_GL = "./training_files/" + Language.GL.value + "_training-tweets.txt"
fileName_ES = "./training_files/" + Language.ES.value + "_training-tweets.txt"
fileName_EN = "./training_files/" + Language.EN.value + "_training-tweets.txt"
fileName_PT = "./training_files/" + Language.PT.value + "_training-tweets.txt"

fEU = open(fileName_EU, "w")
fCA = open(fileName_CA, "w")
fGL = open(fileName_GL, "w")
fES = open(fileName_ES, "w")
fEN = open(fileName_EN, "w")
fPT = open(fileName_PT, "w")

with open("./OriginalDataSet/training-tweets.txt", "r") as file:
    for line in file:
        tweetArray = line.split("\t")
        if tweetArray[2] == "eu":
            fEU.write(line)
        if tweetArray[2] == "ca":
            fCA.write(line)
        if tweetArray[2] == "gl":
            fGL.write(line)
        if tweetArray[2] == "es":
            fES.write(line)
        if tweetArray[2] == "en":
            fEN.write(line)
        if tweetArray[2] == "pt":
            fPT.write(line)
    fEU.close()
    fCA.close()
    fGL.close()
    fES.close()
    fEN.close()
    fPT.close()
