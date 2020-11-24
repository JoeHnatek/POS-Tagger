"""
Written by: Joseph Hnatek
Date: Nov. 23, 2020

Overall:
This program will evaluate the trained and tested data with a gold standard.
When done, it will output the results to a file.

Example:
python3 tagger-eval.py ./PA5-wsj-pos/pos-key.txt pos-test-0.txt
-- OUTPUT TO FILE --
Accuracy: 92.03%
# #: 5
$ $: 375
'' '': 528
( (: 76
) ): 76
, ,: 3070
. .: 2363
: :: 336
CC CC: 1361

python3 tagger-eval.py ./PA5-wsj-pos/pos-key.txt pos-test-1.txt
-- OUTPUT TO FILE --
Accuracy: 93.04%
# #: 5
$ $: 373
'' '': 528
( (: 76
) ): 76
, ,: 3068
. .: 2360
: :: 271
CC CC: 1361

Algorithm:
First, pre-process the given gold standard data.
Next, pre-process the given user/computer data.
Next, we tokenize both the gold standard and user/computer data for better handling.
Next, we compare the results of the gold standard to the user/computer.
Lastly, we output the results of compare to a file.
"""

import sys
import re

def preProcess(path):
    """
    preProcessData will take in a file and read it to a variable, where we will then
    split each line for easy tokenization. We remove the last entry due to it being an empty space.
    """
    with open(path) as filename:
        data = filename.read()

    data = data.split('\n')

    del data[-1]

    return data

def tokenize(data):
    """
    Tokenize will append words and their tags to a list for easier handling of the data for comparing.
    """
    result = []

    for line in data:

        line = re.sub(r'(\\/)', r' ', line)
        line = line.split('/')

        if(" " in line[0]):
            line[0] = line[0].replace(" ", "/")

        word = line[0]
        tag = line[1]

        result.append((word, tag))
    return result

def compare(golden, user):
    """
    Compare will compare the results from the gold standard to the user/computer standard.
    """

    confusionMatrix = {}

    count = 0
    right = 0

    for i in range(len(golden)):
        #print(user)
        goldenWord = golden[i][0]
        goldenTag = golden[i][1]
        userWord = user[i][0]
        userTag = user[i][1]

        if userTag not in confusionMatrix:  # If the users tag is not in the matrix, add it.
            confusionMatrix[userTag] = {}
        
        if goldenTag not in confusionMatrix[userTag]:   # If the gold tag not in the matrix, set it to 1
            confusionMatrix[userTag][goldenTag] = 1
        else:
            confusionMatrix[userTag][goldenTag] += 1    # If the gold is in the matrix, add one to the user tag.


        if goldenTag == userTag:
            right += 1

        count += 1
    accuracy = (right/count)*100    # Give the accuracy.

    return accuracy, confusionMatrix

def sortedOutput(filename, matrix):
    """
    Sorted Output will output the compared results in sorted order of Alpha ascending.
    """
    sortedVals = []
    sortedKeys = sorted(matrix.keys())  # Sort the user tags. (predict tags)
    for key in sortedKeys:
        sortedVals = sorted(matrix[key].items())    # Sort the gold tags
        for val in sortedVals:
            goldTag = val[0]
            count = val[1]

            filename.write("{} {}: {}\n".format(key, goldTag, count))

        sortedVals = []

def output(accuracy, matrix, mode):
    """
    Output will write to a eval file.
    """
    
    if mode == 0:
        with open("pos-test-0-eval.txt", "w") as filename:
            filename.write("Accuracy: {:.2f}%\n".format(accuracy))
            sortedOutput(filename, matrix)

    elif mode == 1:
        with open("pos-test-1-eval.txt", "w") as filename:
            filename.write("Accuracy: {:.2f}%\n".format(accuracy))
            sortedOutput(filename, matrix)
    else:
        exit()

def main(keyPath, myTestPath, mode):
    golden = preProcess(keyPath)
    user = preProcess(myTestPath)
    goldenReady = tokenize(golden)
    userReady = tokenize(user)
    result, matrix = compare(goldenReady, userReady)
    output(result, matrix, mode)

if __name__ == "__main__":
    keyPath = sys.argv[1]
    myTestPath = sys.argv[2]

    if "0" in myTestPath:
        mode = 0
    elif "1" in myTestPath:
        mode = 1
    else:
        exit()
    main(keyPath, myTestPath, mode)