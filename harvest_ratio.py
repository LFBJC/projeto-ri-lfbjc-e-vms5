# -*- coding: utf-8 -*-
import codecs
import glob
from time import time

import matplotlib
import nltk
import numpy
import requests
from selectolax.parser import HTMLParser
try: #python3
    from urllib.request import urlopen, Request
except: #python2
    from urllib2 import urlopen, Request
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
def get_text(soup):
    #kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def tokenize(sentences):
    words = []
    for sentence in sentences:
        print(sentence)
        w = word_extraction(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def word_extraction(sentence):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    #words = re.sub("[^\w]", " ", sentence).split()
    #words = word_tokenize(sentence, 'portuguese')
    words = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', sentence).split()
    #words = words.split('[A-Z][^A-Z]*')
    cleaned_text = [w.lower() for w in words if w not in stopwords]
    return cleaned_text
def bag_of_words_from_sentences(allsentences,vocab):
    bag_of_words = []
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = [0] * len(vocab)
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
        bag_of_words.append(bag_vector)
        return bag_of_words
#pegas as urls da tabela e transformas os htmls em strings
tabela = 'Rotulos_sites.csv'
dataset = pandas.read_csv(tabela)
allsentences = []
for url in dataset['Página']:
 try:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    print('Funcionou: ', url)
    sopa = BeautifulSoup(webpage, 'html.parser')
    texto = get_text(sopa)
    allsentences.append(texto)
 except:
    print(url)
print("____________________________________saiu___________________________________")

#tokenizacao e remocao de stopwords
vocab = tokenize(allsentences)
print("Word List for Document \n{0} \n".format(vocab))

#vectorizer = CountVectorizer()
#bag_of_words = vectorizer.fit(allsentences)
#bag_of_words = vectorizer.transform(allsentences)
#print(bag_of_words)

print("saiu")
#bag_of_words.insert(0, vocab)
#print(bag_of_words)
#print(len(bag_of_words))

array = dataset.values
y = array[:, 2]
y=y.astype('int')
X = bag_of_words_from_sentences(allsentences,vocab)
#X = X.astype('int')
#print(X)
print("NÚMERO DE FEATURES PRÉ SELEÇÃO DE FEATURES:", len(X[0]))


X_new = SelectKBest(chi2, k=15).fit_transform(X, y)
print("NÚMERO DE FEATURES PÓS SELEÇÃO DE FEATURES:", len(X_new[0]))
X_train, X_test, Y_train, Y_test = train_test_split(X_new, y, test_size = 0.25, random_state = 10)

#criando e treinando arvore de decisao
tree = tree.DecisionTreeClassifier(criterion='entropy', random_state=1)
tree = tree.fit(X_train, Y_train)
def predictions(tree,path):
    os.chdir(path)
    allsentences = []
    for file in glob.glob("*.html"):
        sopa = BeautifulSoup(codecs.open(file),'html.parser')
        texto = get_text(sopa)
        allsentences.append(texto)
    vocab = tokenize(allsentences)
    X_baseline = bag_of_words_from_sentences(allsentences,vocab)
    return tree.predict(X_baseline)
baseline_predictions = predictions(tree, "C:\\Users\\naovi\\Desktop\\projeto-ri-lfbjc-e-vms5\\sites_baixados")
heuristica_predictions = predictions(tree, "C:\\Users\\naovi\\Desktop\\projeto-ri-lfbjc-e-vms5\\sites_com_heuristica")
print("harvest ratio do baseline: ", sum(v for v in baseline_predictions if v == 1)/len(baseline_predictions))
print("harvest ratio da heuristica: ", sum(v for v in heuristica_predictions if v == 1)/len(heuristica_predictions))
