"""
TODO
rankear usando modelo de espaço de vetores com e sem tf idf
calc correlação dos ranks
"""
import pandas as pd
import json
query = raw_input('digite aqui os termos da consulta: ')
query_terms = query.split()
with open('index.json') as json_file:
    data = json.load(json_file)
docs = []
for t in query_terms:
	docs.append(data[t])
filenames = []
file_data_base = pd.read_csv('Rotulos_sites.csv')
urls = []
for d in docs:
	urls.append(file_data_base.iloc[1,d[0]])
#tratamento da busca avançada
tipo_de_busca = raw_input('digite \'a\' para busca avançada ou \'c\' para busca comum: ')
qtos_de = 0
qtos_ate = 4294967296
banheiros_de = 0
banheiros_ate = 4294967296
area_de = 0
area_ate = 4294967296
preco_de = 0
preco_ate = 4294967296
if (tipo_de_busca == 'a'):
	qtos_de = input('digite o número mínimo de quartos: ')
	qtos_ate = input('digite o número máximo de quartos: ')
	banheiros_de = input('digite o número mínimo de banheiros: ')
	banheiros_ate = input('digite o número máximo de banheiros: ')
	area_de = input('digite a área mínima: ')
	area_ate = input('digite a área máxima: ')
	preco_de = input('digite o preço mínimo: ')
	preco_ate = input('digite o preço máximo: ')
