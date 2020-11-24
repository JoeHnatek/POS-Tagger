"""
Written by: Joseph Hnatek
Date: Nov. 23, 2020

Overall:
This program will take in a mode, our training data, and the testing data.
It will then use our trained data to create a Part-Of-Speech tagger.
When completed, it will output the results to  'pos-test-x.txt', x depending on the mode.

Example:
python3 tagger-test.py 0 tagger-train-prob.txt ./PA5-wsj-pos/pos-test.txt
-- OUTPUT TO FILE --
No/DT
,/,
it/PRP
was/VBD
n't/RB
Black/NNP
Monday/NNP
./.
But/CC
while/IN


python3 tagger-test.py 1 tagger-train-prob.txt ./PA5-wsj-pos/pos-test.txt
-- OUTPUT TO FILE --
No/DT
,/,
it/PRP
was/VBD
n't/RB
Black/NNP
Monday/NNP
./.
But/CC
while/IN

Algorithm:
First, pre-preprocess the given trained data file.
Next, we tokenize the data so that we can bundle the word - given tag and probability, for better handling when we compare.
Next, we get all the unique words from the data for easy handling and comparing.
Next, we classify the word with a POS tag.
    Mode 0:
        1. If we know the word.
            a. Give the word its most frequent tag.
        2. If we don't know the word.
            a. Give it the tag NN
    Mode 1:
        1. If we know the word.
            a. Check if the word contains a hypen, if so, give it the tag JJ
            b. Check if the previous tag was TO or RB and if so, if the current tag is NN, give it the tag VB instead.
            c. Check if the previous tag was NNP, if so, check the current tag, if it is NNPS, give it the tag NNP instead.
            d. Check if the previuos was was 'have, be, or has', if so, give it the tag VBN.
            e. Check if the word is a given time, ex) '09:00', if so, give it the tag CD.
            f. If we do not match the above cases, give it the most frequent tag.
        2. If we don't know the word.
            a. Check if the first letter in the word is uppercase, give it the tag NNP.
            b. Check if the word contains a hypen, if so, give it the tag JJ.
            c. Check if the word ends in '-ing', if so, give it the tag VBG.
            d. Check if the word ends in '-ed', if so, give it the tag VBD.
            e. Check if the word is a given time, ex) '09:00', if so, give it the tag CD.
            f. If we do not match the above cases, give it the most frequent tag.
Lastly, output the results from our classifier.
"""

import sys
import re

def preProcess(trainDataFile):
    """
    preProcess will process the data before we start training the data.
    """ 

    with open(trainDataFile) as file:   # Open the training data file.
        data = file.read()

    data = data.split('\n') # Split each line on \n

    del data[-1]    # Remove the last entry as it will be a empty space.

    return data

def tokenize(data):
    """
    Tokenize will create our 'easy-to-handle' data structure, so that we can get the word, tag, and probability.
    Structure: {WORD: {.9324: TAG}, ...}
    """
    result = {}

    for line in data:
        line = re.sub(r'(\+)', r' ', line)
        line = line.split()
        word = line[0]
        tag = line[1]
        prob = float(line[2])

        if word not in result:
            result[word] = {}

        if prob not in result[word]:
            result[word][prob] = [tag]
        else:
            result[word][prob].append(tag)

    return result

def getWords(testDataFile):
    """
    Get all of the words from the data.
    """

    data = preProcess(testDataFile)
    
    return data

def classify(words, trainedDict, mode):
    """
    Classify will give a word its tag depending on the mode and rules A1-A5 and B1-B5
    """

    if mode == 0:

        result = []
        
        for word in words:
            if word in trainedDict:
                tag = trainedDict[word][max(trainedDict[word])][0]  # Most frequent tag.
            else:
                tag = "NN"

            result.append("{}/{}".format(word, tag))

    elif mode == 1:
        
        result = []
        checkFor = ["have", "be", "has"]
        prevTag = None
        prevWord = None
        for word in words:

            if word in trainedDict:
                if "-" in word and "--" != word:    # B1 Check if the word contains a hypen, if so, give it the tag JJ
                    tag = "JJ"
                    result.append("{}/{}".format(word, tag))
                elif prevTag == "TO" or prevTag == "RB":    # B2 Check if the previous tag was TO or RB and if so, if the current tag is NN, give it the tag VB instead.
                    tag = trainedDict[word][max(trainedDict[word])][0]
                    if tag == "NN":
                        tag = "VB"
                        result.append("{}/{}".format(word, tag))
                    else:
                        result.append("{}/{}".format(word, tag))
                elif prevTag == "NNP":  # B3 Check if the previous tag was NNP, if so, check the current tag, if it is NNPS, give it the tag NNP instead.
                    tag = trainedDict[word][max(trainedDict[word])][0]
                    if tag == "NNPS":
                        tag = "NNP"
                        result.append("{}/{}".format(word, tag))
                    else:
                        result.append("{}/{}".format(word, tag))
                elif prevWord in checkFor:  # B4 Check if the previuos was was 'have, be, or has', if so, give it the tag VBN.
                    tag = "VBN"
                    result.append("{}/{}".format(word, tag))
                elif re.match(r'[0-9:0-9]+', word): # B5 Check if the word is a given time, ex) '09:00', if so, give it the tag CD.
                    tag = "CD"
                    result.append("{}/{}".format(word, tag))
                else:   # If we do not match the above cases, give it the most frequent tag.
                    tag = trainedDict[word][max(trainedDict[word])][0]
                    result.append("{}/{}".format(word, tag))
                    
            else:
                if word[0].isupper():    # A1 Check if the first letter in the word is uppercase, give it the tag NNP.
                    #print(word[0], word, True)
                    tag = "NNP"
                    result.append("{}/{}".format(word, tag))
                elif "-" in word and "--" != word:  # A2 Check if the word contains a hypen, if so, give it the tag JJ.
                    tag = "JJ"
                    result.append("{}/{}".format(word, tag))
                elif "ing" == word[-3:]:    # A3 Check if the word ends in '-ing', if so, give it the tag VBG.
                    tag = "VBG"
                    result.append("{}/{}".format(word, tag))
                elif "ed" == word[-2:]: # A4 Check if the word ends in '-ed', if so, give it the tag VBD.
                    tag = "VBD"
                    result.append("{}/{}".format(word, tag))
                elif re.match(r'[0-9:0-9]+', word): # A5 Check if the word is a given time, ex) '09:00', if so, give it the tag CD.
                    tag = "CD"
                    result.append("{}/{}".format(word, tag))
                else: # If we do not match the above cases, give it NN.
                    tag = "NN"
                    result.append("{}/{}".format(word, tag))

            prevTag = tag
            prevWord = word
    else:
        print("How did you get to here??")
        print("mode should be 0 or 1")
        exit()    

    return result

def output(data, mode):

    if mode == 0:
        path = "pos-test-0.txt"
    elif mode == 1:
        path = "pos-test-1.txt"
    else:
        exit()

    with open(path, "w") as filename:   # Output the data to the eval file.
        for pair in data:
            filename.write("{}\n".format(pair))

def main(mode, trainDataFile, testDataFile):

    data = preProcess(trainDataFile)
    trainedDict = tokenize(data)
    words = getWords(testDataFile)
    result = classify(words, trainedDict, mode)
    
    output(result, mode)

if __name__ == "__main__":
    mode = int(sys.argv[1])
    trainDataFile = sys.argv[2]
    testDataFile = sys.argv[3]
    main(mode, trainDataFile, testDataFile)