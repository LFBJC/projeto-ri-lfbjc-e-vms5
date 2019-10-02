import nltk
import numpy
import requests
from selectolax.parser import HTMLParser
from urllib.request import Request, urlopen
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

#pegas as urls da tabela e transformas os htmls em strings
tabela = 'Rotulos_sites.csv'
dataset = pandas.read_csv(tabela)
allsentences = []
for url in dataset['Página']:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    print('Funcionou: ', url)
    sopa = BeautifulSoup(webpage, 'html.parser')
    texto = get_text(sopa)
    allsentences.append(texto)
print("____________________________________saiu___________________________________")

#tokenizacao e remocao de stopwords
vocab = tokenize(allsentences)
print("Word List for Document \n{0} \n".format(vocab))

#vectorizer = CountVectorizer()
#bag_of_words = vectorizer.fit(allsentences)
#bag_of_words = vectorizer.transform(allsentences)
#print(bag_of_words)


bag_of_words = []
for sentence in allsentences:
    words = word_extraction(sentence)
    bag_vector = [0] * len(vocab)
    for w in words:
        for i, word in enumerate(vocab):
            if word == w:
                bag_vector[i] += 1
    #print("{0} \n{1}\n".format(sentence, numpy.array(bag_vector)))
    bag_of_words.append(bag_vector)

print("saiu")
#bag_of_words.insert(0, vocab)
#print(bag_of_words)
#print(len(bag_of_words))

array = dataset.values
y = array[:, 2]
y=y.astype('int')
X = bag_of_words
#X = X.astype('int')
#print(X)
print(len(X[0]))
print(len(X[5]))
print(len(X[13]))
print(len(X[22]))

print(y)
print(len(y))

X_new = SelectKBest(chi2, k=15).fit_transform(X, y)
print(len(X_new))
print(len(X_new[0]))
print(len(X_new[5]))
print(len(X_new[13]))
print(len(X_new[22]))

#usando o metodo para faer uma unica divisao dos dados
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

#criando arvore de decisao
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=1)

#treinando a arvore
clf = clf.fit(X_train, Y_train)

t1 = round(clf.score(X_train, Y_train), 3)

te1 = round(clf.score(X_test, Y_test), 3)

print("Acuracia de treinamento clf: %0.3f" % t1)
print("Acuracia de teste clf: %0.3f" % te1)

#texto = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', texto)

#lista = texto.split()
#lista = texto.split('[A-Z][^A-Z]*')
#print(lista)

#lista = re.findall('[A-ZÁÉÍÓÚÂÊÎÔÃÕÇ]+[a-záéíóúâêîôãõç]*|[A-ZÁÉÍÓÚÂÊÎÔÃÕÇ]*[a-záéíóúâêîôãõç]+', texto)

#print(lista)
#print(listanova)