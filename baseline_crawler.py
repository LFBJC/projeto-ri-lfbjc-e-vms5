"""TODO
ALGO ERRADO NAS REQUISIÇÕES (NÃO QUER NEM PRINTAR O CÓDIGO DE STATUS, Nem mesmo try except)
TRATAR CASO DE HAVEREM OUTROS AGENTES DEPOIS D *"""
###########################################################################################################
#  Imports
import requests
import re
from bs4 import BeautifulSoup
from threading import Thread
############################################################################################################
#  Função visit
def visit(url,visited_URLs,to_be_visited,contents,forbidden):
  protocol, url_wout_protocol = url.split('://')
  pure_site_url=''
  if(url_wout_protocol.find('/')!=-1):
    pure_site_url = protocol+'://'+url_wout_protocol[0:url_wout_protocol.find('/')+1]
  else:
    pure_site_url = protocol+'://'+url_wout_protocol+'/'
  URL_robots = pure_site_url + "robots.txt"
  URL_node = url
  visited_URLs.append(URL_node);
  filter(lambda a: a != URL_node, to_be_visited)
  try:
    req_robots = requests.get(url = URL_robots)
  except name_error:
    print('houve um erro '+name_error)
  print(URL_robots)
  robots = req_robots.content.decode('utf-8')
  print(URL_robots)
  matchRobots = re.search('User-agent: \*.*',robots)
  if(matchRobots == None):
    print('o site ' + URL_robots + ' deu erro!!!!!!!!!!!!!')
    return (visited_URLs,to_be_visited,contents,forbidden)
  locally_forbidden = [s[re.search('Disallow: ',s).end():] for s in re.findall('Disallow: .*',robots[matchRobots.start():])]
  locally_forbidden = map(lambda s: pure_site_url[:-1]+s, locally_forbidden)
  forbidden.extend(locally_forbidden)
  node_content = requests.get(url = URL_node).content
  f = open('sites baixados/' + URL_node + '.html','w+')
  contents.append(node_content)
  return (visited_URLs,to_be_visited,contents,forbidden)
############################################################################################################
#  Função que acha a vizinhança
def find_neibourhood(url,to_be_visited,contents,forbidden):
  print('encontrando vizinhança de '+ url)
  soup = BeautifulSoup(contents[-1])
  for anchor in soup.findAll('a'):
    if('href' in anchor.attrs): #checa se realmente é um link pq algumas ancoras estao sem links ou chamam scripts
      href = anchor.attrs['href']
      if((href[:href.find(':')]=='http') or (href[:href.find(':')]=='https')):
         to_be_visited.append(href)
    filter(lambda l: all(l!=href for href in visited_URLs), to_be_visited);#retira endereços visitados
    filter(lambda href: all(href!=s for s in forbidden),to_be_visited);#retira endereços proibidos
    return (to_be_visited,contents,forbidden)
############################################################################################################
#  Função de entrada do baseline
def baseline(url,visited_URLs,to_be_visited,contents,forbidden):
  visited_URLs,to_be_visited,contents,forbidden = visit(url,visited_URLs,to_be_visited,contents,forbidden);
  to_be_visited,contents,forbidden = find_neibourhood(url,to_be_visited,contents,forbidden)
  for ref in to_be_visited:
    visited_URLs,to_be_visited,contents,forbidden = visit(ref,visited_URLs,to_be_visited,contents,forbidden);
    to_be_visited,contents,forbidden = find_neibourhood(ref,to_be_visited,contents,forbidden)
    print(to_be_visited)
    time.sleep(10)
############################################################################################################
#  Definição das threads
class Th(Thread):
  def __init__ (self, url):
                      Thread.__init__(self)
                      self.url = url
  def run(self):
    baseline(url=self.url,visited_URLs=[],to_be_visited = [],contents = [],forbidden = [])
############################################################################################################
#  Parte principal do programa
URLs = ["https://pe.olx.com.br/imoveis","http://www.expoimovel.com/recife/", "https://www.vivareal.com.br/",
        "https://www.imovelweb.com.br/","https://www.chavesnamao.com.br/","https://www.trueimoveis.com.br/imoveis/",
        "https://www.newville.com.br/imoveis/","https://imoveis.trovit.com.br/","https://www.mercadolivre.com.br/imoveis",
        "https://ancoraimobiliaria.com.br/", "https://www.zapimoveis.com.br/","https://apsa.com.br/imoveis",
        "https://www.paulomiranda.com.br/imoveis/","https://www.gedeaoimoveis.com.br/"];
for url in URLs:
#url = URLs[0]
  th = Th(url);
  th.start();