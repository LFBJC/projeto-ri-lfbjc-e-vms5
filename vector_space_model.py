# -*- coding: utf-8 -*-
import nltk
from selectolax.parser import HTMLParser
#fazer inicializacao d documentVectors
class vector_space_model:
    def __init__(self,documentList):
        self.parser = HTMLParser
        self.stemmer = PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
        self.documentVectors = self.makeVector(self.getVectorKeywordIndex(documentList))
    def removeStopWords(self,list):
                 """ Remove common words which have no search value """
                 return [word for word in list if word not in self.stopwords ]
    def tokenise(self, string):
                 """ break string up into tokens and stem words """
                 string = self.clean(string)
                 words = string.split(" ")
                 return [self.stemmer.stem(word,0,len(word)-1) for word in words]
    def getVectorKeywordIndex(self, documentList):
            """ create the keyword associated to the position of the elements within the document vectors """
    
            #Mapped documents into a single word string
            vocabularyString = " ".join(documentList)

            vocabularyList = self.parser.tokenise(vocabularyString)
            #Remove common words which have no search value
            vocabularyList = self.parser.removeStopWords(vocabularyList)
            uniqueVocabularyList = util.removeDuplicates(vocabularyList)
    
            vectorIndex={}
            offset=0
            #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
            for word in uniqueVocabularyList:
                    vectorIndex[word]=offset
                    offset+=1
            return vectorIndex  #(keyword:position)
    def makeVector(self, wordString):
            """ @pre: unique(vectorIndex) """
    
            #Initialise vector with 0's
            vector = [0] * len(self.vectorKeywordIndex)
            wordList = self.parser.tokenise(wordString)
            wordList = self.parser.removeStopWords(wordList)
            for word in wordList:
               vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
            return vector
    def cosine(vector1, vector2):
            """ related documents j and q are in the concept space by comparing the vectors :
                    cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
            return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))
    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        ratings.sort(reverse=True)
        return ratings