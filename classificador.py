import requests
from selectolax.parser import HTMLParser
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re

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

url = 'https://revista.zapimoveis.com.br/?utm_source=zapimoveis&utm_medium=link-header&utm_campaign=btn-zapemcasa'
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()
print(webpage)
sopa = BeautifulSoup(webpage, 'html.parser')
texto = get_text(sopa)
#string_nova = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', texto)
texto = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', texto)

print(texto)
lista = texto.split()
#lista = texto.split('[A-Z][^A-Z]*')
print(lista)

lista = re.findall('[A-ZÁÉÍÓÚÂÊÎÔÃÕÇ]+[a-záéíóúâêîôãõç]*|[A-ZÁÉÍÓÚÂÊÎÔÃÕÇ]*[a-záéíóúâêîôãõç]+', texto)

print(lista)
#print(listanova)