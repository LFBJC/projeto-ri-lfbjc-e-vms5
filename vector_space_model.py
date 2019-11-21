# -*- coding: utf-8 -*-
import nltk
from bs4 import BeautifulSoup
import collections
import re
from sklearn.feature_extraction.text import CountVectorizer

class vector_space_model:
    def __init__(self):
        self.stemmer = nltk.stem.PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
    
    def word_extraction(self, sentence):
        stopwords = nltk.corpus.stopwords.words('portuguese')
        #words = re.sub("[^\w]", " ", sentence).split()
        #words = word_tokenize(sentence, 'portuguese')
        words = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', sentence).split()
        #words = words.split('[A-Z][^A-Z]*')
        cleaned_text = [w.lower() for w in words if w not in stopwords]
        return cleaned_text
    def get_vectors(self, dictionary):
            """ @pre: unique(vectorIndex) """
            vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'))
            Z = vectorizer.fit_transform(dictionary.values())
            vocab = vectorizer.get_feature_names()
            vectorizer.fit_transform(allsentences)
            print(vocab)
            vectors = dict()
            for (k,sentence) in dictionary.items():
                words = self.word_extraction(sentence)
                vectors[k] = [0] * len(vocab)
            for w in words:
                for i, word in enumerate(vocab):
                   if word == w:
                            vectors[k][i] += 1
            return vectors
    def cosine(vector1, vector2):
            """ related documents j and q are in the concept space by comparing the vectors :
                    cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
            return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))
    def rank(self,query,dictionary):
        """ search for documents that match based on a list of terms """
        dictionary['query']=query
        vectors = self.get_vectors(dictionary)
        queryVector = vectors['query']
        del vectors['query']
        ratings = {{k: self.cossine(v,queryVector) for k, v in vectors.items()}}
        ratings = colletions.OrderedDict(sorted(ratings.items(), key=operator.itemgetter(1),reverse=True))
        return ratings
    def rank_tf_idf(self,query,dictionary):
        """ search for documents that match based on a list of terms """
        dictionary['query']=query
        vectors = self.get_vectors(dictionary)
        queryVector = vectors['query']
        del vectors['query']
        vectors = tf_idf_vectors(vectors)
        ratings = {{k: self.cossine(v,queryVector) for k, v in vectors.items()}}
        ratings = colletions.OrderedDict(sorted(ratings.items(), key=operator.itemgetter(1),reverse=True))
        return ratings
    def tf_idf_vectors(vectors,queryVector):
        idf = create_idf_vector(vectors)
        vectors = {{k: [t*idf for t in v] for k, v in vectors.items()}}
        return vectors
    def create_idf_vector(vectors):
        