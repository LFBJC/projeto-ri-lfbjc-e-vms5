# -*- coding: utf-8 -*-
"""TODO
Corrigir 'NoneType' object is not iterable na função find_neighbourhood
Tratar problemas no robots (aparentemente não são apenas strings que completam a url e sim expressões regulares)"""
###########################################################################################################
#  Imports
import requests
import time
import re
from bs4 import BeautifulSoup
from threading import Thread
############################################################################################################
#Setando codificação utf-8
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
############################################################################################################
#  Função que pega o url puro do site
def pure_site_url(url):
  protocol, url_wout_protocol = url.split('://')
  ans=url
  if(url_wout_protocol.find('/')!=-1):
    ans = protocol+'://'+url_wout_protocol[0:url_wout_protocol.find('/')]
  return ans
############################################################################################################
# Checa se qualquer string contém uma das substrings
def any_of_those_substrings(str_list,sub_str_list):
    resp = False
    for stri in str_list:
       resp=resp or any(stri.find(s)!=-1 for s in sub_str_list)
    if(resp == True):
    	print(str_list[0] + " satisfaz a heuristica")
    return resp
############################################################################################################
#  Heuristica
def heuristica(url,anchor_contents):
	str_list=[url]
	if(anchor_contents):
		str_list.extend(anchor_contents)
	return any_of_those_substrings(str_list ,['imov','imob','apartamento','quartos','suíte'])
############################################################################################################
#  Função que identifica as áreas cujo acesso não é permitido
def identify_forbidden_areas(url):
  pure = pure_site_url(url)
  _, file_name = pure.split('://')
  robots=open('arquivos_robots/'+file_name+'.txt','r').read()
  matchRobots = re.search('User-agent: \*.*',robots)
  region_of_interest_for_the_agent = robots[matchRobots.start():]
  other_agents = re.search('User-agent: [^\*]',region_of_interest_for_the_agent)
  if(other_agents):
    region_of_interest_for_the_agent = region_of_interest_for_the_agent[:other_agents.start()]
  forbidden = [s[re.search('Disallow: ',s).end():] for s in re.findall('Disallow: [^\n\r]*',region_of_interest_for_the_agent)]
  forbidden = map(lambda s: pure[:-1]+s, forbidden)
  return forbidden
############################################################################################################
#  Função visit
def visit(url,visited_URLs,to_be_visited,contents):
  protocol, url_wout_protocol = url.split('://')
  visited_URLs.append(url);
  filter(lambda a: a != url, to_be_visited)
  node_content = ''
  if(protocol == 'https'):
    node_content = requests.get(url = url, verify = False).content
  else:
    node_content = requests.get(url = url).content
  f = open('sites_com_heuristica/' + url_wout_protocol.replace('/',' ') + '.html','w+')
  contents.append(node_content)
  return (visited_URLs,to_be_visited,contents)
############################################################################################################
#  Função que acha a vizinhança
def find_neibourhood(url,visited_URLs,to_be_visited,contents,forbidden):
  print('encontrando vizinhança de '+ url)
  on_this_site = pure_site_url(url)
  soup = BeautifulSoup(contents[-1])
  for anchor in soup.findAll('a'):
    if('href' in anchor.attrs): #checa se realmente é um link pq algumas ancoras estao sem links ou chamam scripts
      href = anchor.attrs['href']
      if(re.search(on_this_site,href) and heuristica(href,anchor.text)):
         to_be_visited.append(href)
  to_be_visited = filter(lambda l: all(l!=v for v in visited_URLs), to_be_visited);#retira endereços visitados
  to_be_visited = filter(lambda href: all(href!=s for s in forbidden),to_be_visited);#retira endereços proibidos
  print('sites já visitados: ',visited_URLs)
  print('sites na fronteira: ',to_be_visited)
  return (to_be_visited,contents,forbidden)
############################################################################################################
#  Função de entrada do baseline
def baseline(url,visited_URLs,to_be_visited,contents,forbidden):
  visited_URLs,to_be_visited,contents= visit(url,visited_URLs,to_be_visited,contents);
  to_be_visited,contents,forbidden = find_neibourhood(url,visited_URLs,to_be_visited,contents,forbidden)
  for ref in to_be_visited:
    visited_URLs,to_be_visited,contents = visit(ref,visited_URLs,to_be_visited,contents);
    to_be_visited,contents,forbidden = find_neibourhood(ref,visited_URLs,to_be_visited,contents,forbidden)
    time.sleep(10)
############################################################################################################
#  Definição do funcionamento das threads
class Th(Thread):
  def __init__ (self, url):
                      Thread.__init__(self)
                      self.url = url
  def run(self):
    baseline(url=self.url,visited_URLs=[],to_be_visited = [],contents = [],forbidden = identify_forbidden_areas(url))
############################################################################################################
#  Parte principal do programa
URLs = ["https://pe.olx.com.br/imoveis","http://www.expoimovel.com/recife/", "https://www.vivareal.com.br/",
        "https://www.chavesnamao.com.br/","https://www.trueimoveis.com.br/imoveis/",
        "https://www.newville.com.br/imoveis/","https://imoveis.trovit.com.br/","https://www.mercadolivre.com.br/imoveis",
        "https://ancoraimobiliaria.com.br/", "https://www.zapimoveis.com.br/","https://apsa.com.br/imoveis",
        "https://www.paulomiranda.com.br/imoveis/","https://www.gedeaoimoveis.com.br/"];
for url in URLs:
#url = URLs[0]
  th = Th(url);
  th.start();
