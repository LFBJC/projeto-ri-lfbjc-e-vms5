"""
TODO
rankear usando modelo de espaço de vetores com e sem tf idf
calc correlação dos ranks
"""
import json
query = raw_input('digite aqui os termos da consulta: ')
query_terms = query.split()
with open('index.json') as json_file:
    data = json.load(json_file)
docs = []
for t in query_terms:
	docs.append(data[t])
with open('ordem de documentos.txt') as docs_file:
    files = docs_file.read().replace('\n', '').replace(' ','').replace('[', '').replace(']','').split(',')
filenames = []
for d in docs:
	filenames.append(files[d[0]])
#tratamento da busca avançada
tipo_de_busca = raw_input('digite \'a\' para busca avançada ou \'c\' para busca comum: ')
qtos_de = 0
qtos_ate = 4294967296
banheiros_de = 0
banheiros_ate = 4294967296
preco_de = 0
preco_ate = 4294967296
if (tipo_de_busca == 'a'):
	qtos_de = input('digite o número mínimo de quartos: ')
	qtos_ate = input('digite o número máximo de quartos: ')
	banheiros_de = input('digite o número mínimo de banheiros: ')
	banheiros_ate = input('digite o número máximo de banheiros: ')
	preco_de = input('digite o preço mínimo: ')
	preco_ate = input('digite o preço máximo: ')
elements =[]
for name in filenames:
	with open('.\\info_pages\\'+name+'.json') as element:
		elem_features = json.load(element);
		if((elem_features['Quartos']>qtos_de)&(elem_features['Quartos']<qtos_ate)&(elem_features['Banheiros']>banheiros_de)&(elem_features['Banheiros']<banheiros_ate)&(elem_features['Preco']>preco_de)&(elem_features['Preco']<preco_ate)):
			elements.append(elem_features)
#rankeamento

#exibição da busca
