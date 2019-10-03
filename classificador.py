import matplotlib
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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
print("NÚMERO DE FEATURES PRÉ SELEÇÃO DE FEATURES:", len(X[0]))


X_new = SelectKBest(chi2, k=15).fit_transform(X, y)
print("NÚMERO DE FEATURES PÓS SELEÇÃO DE FEATURES:", len(X_new[0]))


#usando o metodo para faer uma unica divisao dos dados
X_train, X_test, Y_train, Y_test = train_test_split(X_new, y, test_size = 0.25, random_state = 10)

#criando e treinando naive bayes
gnb = GaussianNB()
gnb = gnb.fit(X_train, Y_train)
#acuracia treino e test
t1 = round(gnb.score(X_train, Y_train), 3)
te1 = round(gnb.score(X_test, Y_test), 3)
print("Acuracia de treinamento NAIVE BAYES: %0.3f" % t1)
print("Acuracia de teste NAIVE BAYES: %0.3f" % te1)
#precision NAIVE BAYES
y_pred1 = gnb.predict(X_test)
micro_precision1 = precision_score(Y_test, y_pred1, average='weighted')
print("Precision NAIVE BAYES: %0.3f" % micro_precision1)
#recall NAIVE BAYES
recall1 = recall_score(Y_test, y_pred1, average='weighted')
print("Recall NAIVE BAYES: %0.3f" % recall1)

#criando e treinando arvore de decisao
tree = tree.DecisionTreeClassifier(criterion='entropy', random_state=1)
tree = tree.fit(X_train, Y_train)
t2 = round(tree.score(X_train, Y_train), 3)
te2 = round(tree.score(X_test, Y_test), 3)
#acuracia treino e test
print("Acuracia de treinamento ARVORE DE DECISAO: %0.3f" % t2)
print("Acuracia de teste ARVORE DE DECISAO: %0.3f" % te2)
#precision arvore de decisao
y_pred2 = tree.predict(X_test)
micro_precision2 = precision_score(Y_test, y_pred2, average='weighted')
print("Precision ARVORE DE DECISAO: %0.3f" % micro_precision2)
#recall ARVORE DE DECISAO
recall2 = recall_score(Y_test, y_pred2, average='weighted')
print("Recall ARVORE DE DECISAO: %0.3f" % recall2)

#criando e treinando logistic regression
lr = LogisticRegression(random_state=1)
lr = lr.fit(X_train, Y_train)
#acuracia treino e test
t3 = round(lr.score(X_train, Y_train), 3)
te3 = round(lr.score(X_test, Y_test), 3)
print("Acuracia de treinamento LOGISTIC REGRESSION: %0.3f" % t3)
print("Acuracia de teste LOGISTIC REGRESSION: %0.3f" % te3)
#precision LOGISTIC REGRESSION
y_pred3 = lr.predict(X_test)
micro_precision3 = precision_score(Y_test, y_pred3, average='weighted')
print("Precision LOGISTIC REGRESSION: %0.3f" % micro_precision3)
#recall LOGISTIC REGRESSION
recall3 = recall_score(Y_test, y_pred3, average='weighted')
print("Recall LOGISTIC REGRESSION: %0.3f" % recall3)

#criando e treinando svm
svm = SVC(gamma='auto', random_state=1)
svm.fit(X_train, Y_train)
#acuracia treino e test
t4 = round(svm.score(X_train, Y_train), 3)
te4 = round(svm.score(X_test, Y_test), 3)
print("Acuracia de treinamento SVM: %0.3f" % t4)
print("Acuracia de teste SVM: %0.3f" % te4)
#precision SVM
y_pred4 = svm.predict(X_test)
micro_precision4 = precision_score(Y_test, y_pred4, average='weighted')
print("Precision SVM: %0.3f" % micro_precision4)
#recall SVM
recall4 = recall_score(Y_test, y_pred4, average='weighted')
print("Recall SVM: %0.3f" % recall4)

#criando e treinando mlp
mlp = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=450, random_state=1)
mlp = mlp.fit(X_train, Y_train)
#acuracia treino e test
t5 = round(mlp.score(X_train, Y_train), 3)
te5 = round(mlp.score(X_test, Y_test), 3)
print("Acuracia de treinamento MLP: %0.3f" % t5)
print("Acuracia de teste MLP: %0.3f" % te5)
#precision MLP
y_pred5 = mlp.predict(X_test)
micro_precision5 = precision_score(Y_test, y_pred5, average='weighted')
print("Precision MLP: %0.3f" % micro_precision5)
#recall NAIVE BAYES
recall5 = recall_score(Y_test, y_pred5, average='weighted')
print("Recall MLP: %0.3f" % recall5)

recall_list = [recall1, recall2, recall3, recall4, recall5]
precision_list = [micro_precision1, micro_precision2, micro_precision3, micro_precision4, micro_precision5]

tituloArray = ["NAIVE BAYES", "TREE DECISION", "LOGISTIC", "SVM", "MLP"]
desempenhoTreinoArray = [t1, t2, t3, t4, t5]
desempenhoTestArray = [te1, te2, te3, te4, te5]

matplotlib.pyplot.plot(tituloArray, desempenhoTreinoArray)
matplotlib.pyplot.title('Acuracia treinamento')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(tituloArray, desempenhoTestArray)
matplotlib.pyplot.title('Acuracia teste')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(tituloArray, precision_list)
matplotlib.pyplot.title('Precision teste')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(tituloArray, recall_list)
matplotlib.pyplot.title('Recall teste')
matplotlib.pyplot.show()