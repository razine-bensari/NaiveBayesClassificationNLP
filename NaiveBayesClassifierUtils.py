"""
Author: Razine Ahmed Bensari, bensaria97@gmail.com
Author: Annes Cherrid, annes.cherid@gmail.com

Class with helper function for Naive Bayes classifers.
Helps clean the data and provide insight on the classifier and produce output files required
"""
from enum import Enum
from NBLanguageClassifier import NBLanguageClassifier
import time

class Language(Enum):
    EU = "eu"
    CA = "ca"
    GL = "gl"
    ES = "es"
    EN = "en"
    PT = "pt"


def cleanTrainingData():
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

def trainAllNBLanguageClassifier():
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
                print("\t\t\tIn V = " + str(v) + ", N = " + str(n) + ", D = " + str(
                    d) + ". Time (s) to iterate over D is " + str(round(stopD - startD, 1)))
            stopN = time.time()
            print("\t\tTime (s) to iterate over N = " + str(n) + ", " + str(round(stopN - startN, 1)))
        stopV = time.time()
        print("\tTime (s) to iterate over V = " + str(v) + ", " + str(round(stopV - startV, 1)))
    stopP = time.time()
    print("Time for the whole process: " + str(round(stopP - startP, 1)))


def generateAllEvalFiles():
    for v in range(3):
        for n in range(1, 4):
            for d in range(1, 10):
                delta = d * 0.1
                trace_file_name = "./trace_files/trace_" + str(v) + "_" + str(n) + "_" + str(round(delta, 1)) + ".txt"
                eval_file_name = "./eval_files/eval_" + str(v) + "_" + str(n) + "_" + str(round(delta, 1)) + ".txt"
                f = open(eval_file_name, "w")
                print("Doing this trace file: " + trace_file_name)
                with open(trace_file_name, "r") as file:
                    eu_P = {"correct": 0, "wrong": 0, "actual": 0}
                    ca_P = {"correct": 0, "wrong": 0, "actual": 0}
                    gl_P = {"correct": 0, "wrong": 0, "actual": 0}
                    es_P = {"correct": 0, "wrong": 0, "actual": 0}
                    en_P = {"correct": 0, "wrong": 0, "actual": 0}
                    pt_P = {"correct": 0, "wrong": 0, "actual": 0}
                    num_correct = 0
                    num_wrong = 0
                    for line in file:
                        trace_array = line.split("  ")
                        tweet_id = trace_array[0]
                        most_likely_class = trace_array[1]
                        score = trace_array[2]
                        actual_class = trace_array[3]
                        result = trace_array[4].rstrip()  # correct or wrong
                        if result == "correct":
                            num_correct += 1
                            if most_likely_class == "eu":
                                eu_P["correct"] += 1
                            elif most_likely_class == "ca":
                                ca_P["correct"] += 1
                            elif most_likely_class == "gl":
                                gl_P["correct"] += 1
                            elif most_likely_class == "es":
                                es_P["correct"] += 1
                            elif most_likely_class == "en":
                                en_P["correct"] += 1
                            elif most_likely_class == "pt":
                                pt_P["correct"] += 1
                            else:
                                pass
                            if actual_class == "eu":
                                eu_P["actual"] += 1
                            elif actual_class == "ca":
                                ca_P["actual"] += 1
                            elif actual_class == "gl":
                                gl_P["actual"] += 1
                            elif actual_class == "es":
                                es_P["actual"] += 1
                            elif actual_class == "en":
                                en_P["actual"] += 1
                            elif actual_class == "pt":
                                pt_P["actual"] += 1
                            else:
                                pass
                        else:
                            num_wrong += 1
                            if most_likely_class == "eu":
                                eu_P["wrong"] += 1
                            elif most_likely_class == "ca":
                                ca_P["wrong"] += 1
                            elif most_likely_class == "gl":
                                gl_P["wrong"] += 1
                            elif most_likely_class == "es":
                                es_P["wrong"] += 1
                            elif most_likely_class == "en":
                                en_P["wrong"] += 1
                            elif most_likely_class == "pt":
                                pt_P["wrong"] += 1
                            else:
                                pass
                            if actual_class == "eu":
                                eu_P["actual"] += 1
                            elif actual_class == "ca":
                                ca_P["actual"] += 1
                            elif actual_class == "gl":
                                gl_P["actual"] += 1
                            elif actual_class == "es":
                                es_P["actual"] += 1
                            elif actual_class == "en":
                                en_P["actual"] += 1
                            elif actual_class == "pt":
                                pt_P["actual"] += 1
                            else:
                                pass

                accuracy = round((num_correct / (num_correct + num_wrong)), 4)

                eu_precision = round((eu_P["correct"] / (eu_P["correct"] + eu_P["wrong"])), 4)
                ca_precision = round((ca_P["correct"] / (ca_P["correct"] + ca_P["wrong"])), 4)
                if gl_P["correct"] == 0 and gl_P["wrong"] == 0:
                    gl_precision = .000
                else:
                    gl_precision = round((gl_P["correct"] / (gl_P["correct"] + gl_P["wrong"])), 4)
                es_precision = round((es_P["correct"] / (es_P["correct"] + es_P["wrong"])), 4)
                en_precision = round((en_P["correct"] / (en_P["correct"] + en_P["wrong"])), 4)
                pt_precision = round((pt_P["correct"] / (pt_P["correct"] + pt_P["wrong"])), 4)

                eu_recall = round((eu_P["correct"] / (eu_P["correct"] + eu_P["actual"])), 4)
                ca_recall = round((ca_P["correct"] / (ca_P["correct"] + ca_P["actual"])), 4)
                gl_recall = round((gl_P["correct"] / (gl_P["correct"] + gl_P["actual"])), 4)
                es_recall = round((es_P["correct"] / (es_P["correct"] + es_P["actual"])), 4)
                en_recall = round((en_P["correct"] / (en_P["correct"] + en_P["actual"])), 4)
                pt_recall = round((pt_P["correct"] / (pt_P["correct"] + pt_P["actual"])), 4)

                f.write(str(accuracy) + "\n")
                f.write(str(eu_precision) + "  " + str(ca_precision) + "  " + str(gl_precision) + "  " + str(
                    es_precision) + "  " + str(en_precision) + "  " + str(pt_precision) + "\n")
                f.write(str(eu_recall) + "  " + str(ca_recall) + "  " + str(gl_recall) + "  " + str(
                    es_recall) + "  " + str(en_recall) + "  " + str(pt_recall) + "\n")
                f.close()


def generateStatsFile():
    acc = 0.0
    accv = 0
    accn = 0
    accd = 0.0

    eu_precision = 0
    eu_precisionv = 0
    eu_precisionn = 0
    eu_precisiond = 0.0

    ca_precision = 0
    ca_precisionv = 0
    ca_precisionn = 0
    ca_precisiond = 0.0

    gl_precision = 0
    gl_precisionv = 0
    gl_precisionn = 0
    gl_precisiond = 0.0

    es_precision = 0
    es_precisionv = 0
    es_precisionn = 0
    es_precisiond = 0.0

    en_precision = 0
    en_precisionv = 0
    en_precisionn = 0
    en_precisiond = 0.0

    pt_precision = 0
    pt_precisionv = 0
    pt_precisionn = 0
    pt_precisiond = 0.0

    eu_recall = 0
    eu_recallv = 0
    eu_recalln = 0
    eu_recalld = 0.0

    ca_recall = 0
    ca_recallv = 0
    ca_recalln = 0
    ca_recalld = 0.0

    gl_recall = 0
    gl_recallv = 0
    gl_recalln = 0
    gl_recalld = 0.0

    es_recall = 0
    es_recallv = 0
    es_recalln = 0
    es_recalld = 0.0

    en_recall = 0
    en_recallv = 0
    en_recalln = 0
    en_recalld = 0.0

    pt_recall = 0
    pt_recallv = 0
    pt_recalln = 0
    pt_recalld = 0.0

    for v in range(3):
        for n in range(1, 4):
            for d in range(1, 10):
                delta = d * 0.1
                eval_file_name = "./eval_files/eval_" + str(v) + "_" + str(n) + "_" + str(round(delta, 1)) + ".txt"
                lines = []
                with open(eval_file_name, "r") as file:
                    for line in file:
                        lines.append(line.rstrip())
                if acc < float(lines[0]):
                    acc = float(lines[0])
                    accv = v
                    accn = n
                    accd = round(delta, 4)
                precisions = lines[1].split("  ")
                if eu_precision < float(precisions[0]):
                    eu_precision = float(precisions[0])
                    eu_precisionv = v
                    eu_precisionn = n
                    eu_precisiond = round(delta, 4)
                if ca_precision < float(precisions[1]):
                    ca_precision = float(precisions[1])
                    ca_precisionv = v
                    ca_precisionn = n
                    ca_precisiond = round(delta, 4)
                if gl_precision < float(precisions[2]):
                    gl_precision = float(precisions[2])
                    gl_precisionv = v
                    gl_precisionn = n
                    gl_precisiond = round(delta, 4)
                if es_precision < float(precisions[3]):
                    es_precision = float(precisions[3])
                    es_precisionv = v
                    es_precisionn = n
                    es_precisiond = round(delta, 4)
                if en_precision < float(precisions[4]):
                    en_precision = float(precisions[4])
                    en_precisionv = v
                    en_precisionn = n
                    en_precisiond = round(delta, 4)
                if pt_precision < float(precisions[5]):
                    pt_precision = float(precisions[5])
                    pt_precisionv = v
                    pt_precisionn = n
                    pt_precisiond = round(delta, 4)
                recalls = lines[2].split("  ")
                if eu_recall < float(recalls[0]):
                    eu_recall = float(recalls[0])
                    eu_recallv = v
                    eu_recalln = n
                    eu_recalld = round(delta, 4)
                if ca_recall < float(recalls[1]):
                    ca_recall = float(recalls[1])
                    ca_recallv = v
                    ca_recalln = n
                    ca_recalld = round(delta, 4)
                if gl_recall < float(recalls[2]):
                    gl_recall = float(recalls[2])
                    gl_recallv = v
                    gl_recalln = n
                    gl_recalld = round(delta, 4)
                if es_recall < float(recalls[3]):
                    es_recall = float(recalls[3])
                    es_recallv = v
                    es_recalln = n
                    es_recalld = round(delta, 4)
                if en_recall < float(recalls[4]):
                    en_recall = float(recalls[4])
                    en_recallv = v
                    en_recalln = n
                    en_recalld = round(delta, 4)
                if pt_recall < float(recalls[5]):
                    pt_recall = float(recalls[5])
                    pt_recallv = v
                    pt_recalln = n
                    pt_recalld = round(delta, 4)
    f = open("stats.txt", "w")
    f.write("Some stats regarding the classifier. \n\n")
    f.write("Model with best accuracy: " + str(acc) + " is V = " + str(accv) + ", N = " + str(accn) + " and D = " + str(
        accd) + "\n")
    f.write("\n\n")
    f.write("eu with best precision: " + str(eu_precision) + " is V = " + str(eu_precisionv) + ", N = " + str(
        eu_precisionn) + " and D = " + str(eu_precisiond) + "\n")
    f.write("eu with best recall: " + str(eu_recall) + " is V = " + str(eu_recallv) + ", N = " + str(
        eu_recalln) + " and D = " + str(eu_recalld) + "\n")
    f.write("\n\n")
    f.write("ca with best precision: " + str(ca_precision) + " is V = " + str(ca_precisionv) + ", N = " + str(
        ca_precisionn) + " and D = " + str(ca_precisiond) + "\n")
    f.write("ca with best recall: " + str(ca_recall) + " is V = " + str(ca_recallv) + ", N = " + str(
        ca_recalln) + " and D = " + str(ca_recalld) + "\n")
    f.write("\n\n")
    f.write("gl with best precision: " + str(gl_precision) + " is V = " + str(gl_precisionv) + ", N = " + str(
        gl_precisionn) + " and D = " + str(gl_precisiond) + "\n")
    f.write("gl with best recall: " + str(gl_recall) + " is V = " + str(gl_recallv) + ", N = " + str(
        gl_recalln) + " and D = " + str(gl_recalld) + "\n")
    f.write("\n\n")
    f.write("es with best precision: " + str(es_precision) + " is V = " + str(es_precisionv) + ", N = " + str(
        es_precisionn) + " and D = " + str(es_precisiond) + "\n")
    f.write("es with best recall: " + str(es_recall) + " is V = " + str(es_recallv) + ", N = " + str(
        es_recalln) + " and D = " + str(es_recalld) + "\n")
    f.write("\n\n")
    f.write("en with best precision: " + str(en_precision) + " is V = " + str(en_precisionv) + ", N = " + str(
        en_precisionn) + " and D = " + str(en_precisiond) + "\n")
    f.write("en with best recall: " + str(en_recall) + " is V = " + str(en_recallv) + ", N = " + str(
        en_recalln) + " and D = " + str(en_recalld) + "\n")
    f.write("\n\n")
    f.write("pt with best precision: " + str(pt_precision) + " is V = " + str(pt_precisionv) + ", N = " + str(
        pt_precisionn) + " and D = " + str(pt_precisiond) + "\n")
    f.write("pt with best recall: " + str(pt_recall) + " is V = " + str(pt_recallv) + ", N = " + str(
        pt_recalln) + " and D = " + str(pt_recalld) + "\n")
    f.close()
