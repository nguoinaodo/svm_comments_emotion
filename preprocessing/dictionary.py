import numpy as np 
import re
from stopword import read_stop_words, eliminate_stop_words

def create_dict(docs, stopwords):
	vi_dict = load_vi_dict('vv-dict')
	dic = {}
	count = 0
	reg = re.compile('\s+|\.+|\,+|\b|&|~|"')
	for doc in docs:
		doc = eliminate_stop_words(doc.lower(), stopwords)
		tokens = re.split(reg, doc)
		for t in tokens:
			if dic.has_key(t) == False:# and vi_dict.has_key(t): 
					# and vi_dict.has_key(t) == True:
				dic[t] = count
				count += 1
	return dic

def save_dict(dic, filename):
	with open(filename, 'w') as f:
		for k in dic.keys():
			f.write(k + '&' + str(dic[k]) + '\n')

def read_doc(filename):
	with open(filename) as f:
		s = f.read()
	return s
	
# 4-30162
def read_vi_dict(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()[3: 30162]
		vi_dict = {}
		for l in lines:
			arr = re.split(r"#+", l)
			term = '_'.join(arr[0].split())
			search = re.search(r"\S+\.", arr[2])
			lt = 'unknown'
			if search != None:
				lt = search.group(0)[: -1]
			vi_dict[term] = lt
	return vi_dict

def load_vi_dict(filename):
	vi_dict = {}
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			a = line.split(':')
			vi_dict[a[0]] = a[1].strip()
	return vi_dict	

def load_dict(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		dic = {}
		inv_dic = {}
		for line in lines:
			arr = line.split('&')
			term = arr[0]
			idx = int(arr[1].strip())
			dic[term] = idx
			inv_dic[idx] = term
	return dic, inv_dic	
			
def main():
	# files = ['../data/cleaned/NEGATIVE_TOKENIZED.csv',
	# 		'../data/cleaned/POSITIVE_TOKENIZED.csv',
	# 		# '../data/cleaned/iphone_dev_tokenized',
	# 		'../data/cleaned/iphone_train_tokenized.csv']
	files = ['../data/cleaned/TRAIN.csv']
	docs = []
	for f in files:
		doc = read_doc(f)
		docs.append(doc)
	stopwords = read_stop_words('../data/stopword/vnstopword.txt')
	dic = create_dict(docs, stopwords)
	# vi_dict = read_saved_vi_dict('../data/dictionary/vv-dict')

	save_dict(dic, '../data/dictionary/dictionary')


# main()

# vi_dict = read_vi_dict('../vv/vv.dd')
# save_dict(vi_dict ,'vv-dict')