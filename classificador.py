import requests
from selectolax.parser import HTMLParser
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

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

url = 'https://www.zapimoveis.com.br/informacao?opcao=termouso'
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()
sopa = BeautifulSoup(webpage, 'html.parser')

print(get_text(sopa))