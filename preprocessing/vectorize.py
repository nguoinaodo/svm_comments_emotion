from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from scipy.sparse import coo_matrix, vstack
import pickle
from dictionary import load_dict
from gensim.models import Doc2Vec, Word2Vec

def vectorize_word2vec(trainfile, testfile):
	train = []
	train_labels = []
	test = []
	test_labels = []
	# Train
	train_lines = readlines_split(trainfile)
	for line in train_lines:
		train.append(line[1:])
		train_labels.append(line[0])
	train_labels = np.array(train_labels)
	# Test
	test_lines = readlines_split(testfile)
	for line in test_lines:
		test.append(line[1:])
		test_labels.append(line[0])	
	test_labels = np.array(test_labels)
	# Word2vec
	size = 200
	model = Word2Vec(train, size=size, window=5, min_count=5, workers=4)
	wv = model.wv # word vectors
	# To vector
	train_features = []
	test_features = []
	for s in train:
		s_vec = np.zeros(size)
		count = 0
		for w in s:
			if w in wv.vocab:
				count += 1
				s_vec += wv[w]
		s_vec = 1. * s_vec / count
		train_features.append(s_vec)
	for s in test:
		s_vec = np.zeros(size)
		count = 0
		for w in s:
			if w in wv.vocab:
				count += 1
				s_vec += wv[w]
		s_vec = 1. * s_vec / count
		test_features.append(s_vec)
	train_features = np.array(train_features)
	test_features = np.array(test_features)

	return train_features, train_labels, test_features, test_labels

def readlines(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	return lines

def readlines_split(filename):
	with open(filename, 'r') as f:
		result = []
		lines = f.readlines()
		reg = re.compile('\s+|\.+|\,+|\b|&|~|"')
		for line in lines:
			tokens = re.split(reg, line)
			result.append(tokens)
	return result

# train_features, train_labels, test_features, test_labels = vectorize_word2vec('../data/cleaned/TRAIN.csv',\
# 		'../data/cleaned/TEST.csv')
