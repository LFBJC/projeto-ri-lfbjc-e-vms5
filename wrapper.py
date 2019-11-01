import pandas
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup

tabela = 'sites_wrapper.csv'
dataset = pandas.read_csv(tabela)

'''
allhtmls = []

for url in dataset['Pagina']:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    print('Funcionou: ', url)
    soup = BeautifulSoup(webpage, 'html.parser')
    allhtmls.append(soup)

#escrita dos arquivos
with open('listhtml.txt', 'w') as filehandle:
    for listitem in allhtmls:
        filehandle.write('%s\n' % listitem.encode('utf-8'))
'''
#leitura dos arquivos
allhtmls = []
with open('listhtml.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        allhtmls.append(currentPlace)
        print(currentPlace)
