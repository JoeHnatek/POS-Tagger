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

    return len(data.keys())
        
def computeProb(wordDict, masterTagDict, tagDict, tagVocab):

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

            key = "{}+{}".format(word, tag)

            resultDict[key] = num/den
    
    return resultDict

def output(data):

    with open("tagger-train-prob.txt", "w") as filename:
        for key, value in reversed(sorted(data.items(), key=lambda x: x[1])):
            filename.write("{} {}\n".format(key, value))

def main():

    data = preProcessData(filepath)

    wordDict, tagDict, masterTagDict = tokenize(data)

    tagVocab = getVocabCount(masterTagDict)

    result = computeProb(wordDict, masterTagDict, tagDict, tagVocab)

    output(result)



if __name__ == "__main__":

    filepath = sys.argv[1]  

    main()