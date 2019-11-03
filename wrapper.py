import pandas
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup

#tabela = 'sites_wrapper.csv'
#dataset = pandas.read_csv(tabela)
zap = 'zap.html'
soup = BeautifulSoup(open(zap, encoding='utf-8'), 'html.parser')
#print(get_text(soup))
tag = soup.find('div', {'class':"box--flex-grow"})
print(tag)
