"""
Written by: Joseph Hnatek
Date: Nov. 23, 2020

Overall:
This program will train data given from WSJ articles. 
Once trained, it will output to tagger-train-prob.txt with our trained model from frequency of words.

Example:
python3 tagger-train.py ./PA5-wsj-pos/pos-train.txt
-- OUTPUT TO FILE --
*Note: "word+tag probability"
,+, 0.9992732787765806
.+. 0.9990879530494091
the+DT 0.9988317973631997
of+IN 0.9983161014380494
to+TO 0.9982980792608801
and+CC 0.997352587244284
a+DT 0.9970156476851105
for+IN 0.9950129471564209
$+$ 0.9950086805555556
``+`` 0.994999456462659

Algorithm:
First, pre-process the data from the WSJ articles.
Next, tokenize the pre-processed data so that we have the word and their tag
    1. If the word contains "\/" witin numbers or words.
        a. Remove the "\". Ex) "3\/4" -> "3/4"
Next, get the vocab count of the tags. (Number of tags possible)
Next, compute the probability of P(tag|given).
Finally, output the results to tagger-train-prob.txt
"""


import sys
import re


def preProcessData(filepath):
    """
    preProcessData will take in a file and read it to a variable, where we will then
    split each line for easy tokenization. We remove the last entry due to it being an empty space.
    """
    with open(filepath) as file:    # Read in the training data
        data = file.read()

    data = data.split("\n")

    del data[-1]    # Remove the space at the end

    return data

def tokenize(data):
    """
    Tokenize function will tokenize line by line into words and their tag.
    This function also recogizes the case where "3\/4" is "3/4".
    """

    wordDict = {}
    masterTagDict = {}
    tagDict = {}

    for line in data:
        line = re.sub(r'(\\/)', r' ', line) # If the line contains "\/", sub it with a space
        line = line.split("/")  # Split the data on "/" for word and tag

        if(" " in line[0]): # If there was a "\/" in the word, it is now a space.
            line[0] = line[0].replace(" ", "/") # Replace the space with "/", to fix the word

        word = line[0]
        tag = line[1]

        # Add the word to the master dict
        if word not in wordDict:
            wordDict[word] = 1    # Set the count of word to 1
        else:
            wordDict[word] += 1   # Add one to the count of word
        
        if tag not in masterTagDict:
            masterTagDict[tag] = 1
        else:
            masterTagDict[tag] += 1
        
        # Add the word to a tag.
        if word not in tagDict:
            tagDict[word] = {}
        
        # Add the word to the tag
        if tag not in tagDict[word]:
            tagDict[word][tag] = 1    # Set the count of tag and word to 1
        else:
            tagDict[word][tag] += 1   # Add one to the count of tag and word

    return wordDict, tagDict, masterTagDict

def getVocabCount(data):
    """
    getVocabCount returns the number of unique keys within a master dictionary.
    """
    return len(data.keys())
        
def computeProb(wordDict, masterTagDict, tagDict, tagVocab):
    """
    ComputeProb returns a dictionary containing the resulting P(tag|word).
    """
    resultDict = {}

    for word in wordDict.keys():
        for tag in masterTagDict.keys():
            
            # Calculate numerator of P(tag|word)
            if tag in tagDict[word]:
                num = tagDict[word][tag] + 1
            else:
                num = 1
            
            # Calculate denominator of P(tag|word)
            value = 0
            for tag1 in masterTagDict.keys():
                if tag1 in tagDict[word]:
                    value += tagDict[word][tag1]
            
            den = value + tagVocab

            key = "{}+{}".format(word, tag) # Format the word as "word+tag", so that we can process data better in tester

            resultDict[key] = num/den
    
    return resultDict

def output(data):

    # Write the results of P(tag|word) to the file in order for easy viewing.
    with open("tagger-train-prob.txt", "w") as filename:
        for key, value in reversed(sorted(data.items(), key=lambda x: x[1])):
            filename.write("{} {}\n".format(key, value))

def main():

    data = preProcessData(filepath) # Pre-process our data

    wordDict, tagDict, masterTagDict = tokenize(data)   # Tokenize our data for easy handling.

    tagVocab = getVocabCount(masterTagDict) # Grab the vocab count of tags within the corpus.

    result = computeProb(wordDict, masterTagDict, tagDict, tagVocab)    # Compute P(tag|word)

    output(result)  # Output the results of computeProb to a file.



if __name__ == "__main__":

    filepath = sys.argv[1]  # Grab the test data.

    main()  # Run the program