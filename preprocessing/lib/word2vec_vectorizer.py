from gensim.models import Doc2Vec, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import re
import pickle

"""
Doc2Vec : you can train your dataset using Doc2Vec and then use the sentence vectors.
Average of Word2Vec vectors : You can just take the average of all the word vectors in a sentence. 
		This average vector will represent your sentence vector.
Average of Word2Vec vectors with TF-IDF : this is one of the best approach which I will recommend.
		Just take the word vectors and multiply it with their TF-IDF scores. 
		Just take the average and it will represent your sentence vector.
"""

# Split each line into tokens array
def split_lines(lines):
	result = []
	reg = r'\s+|\.+|\,+|\b|&|~|"'
	for line in lines:
		tokens = re.split(reg, line)
		result.append(tokens)
	return result

# Save pkl file
def save_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

# Load pkl file
def load_pickle(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj

class AverageWord2Vec:
	def __init__(self, size=100, window=5, min_count=5, workers=4, w2vModel=None):
		self.size = size
		self.window = window
		self.min_count = min_count
		self.workers = workers
		self.w2vModel = w2vModel
		if w2vModel:
			self.word_vectors = w2vModel.wv

	def info(self):		
		return 'avg-w2v-size%d-min_count%d' % (self.size, self.min_count)

	def fit(self, lines):
		# Split lines to sentences
		sentences = split_lines(lines)
		# Fit model
		self.w2vModel = Word2Vec(sentences, size=self.size, window=self.window, \
				min_count=self.min_count, workers=self.workers)
		self.word_vectors = self.w2vModel.wv	

	def transform(self, lines):
		result = []
		# Split lines to sentences
		sentences = split_lines(lines)
		for s in sentences:
			vec = np.zeros(self.size)
			count = 0
			for token in s:
				try:
					vec += self.word_vectors[token]
					count += 1
				except KeyError:
					continue
			if count != 0:
				vec = vec / count
			result.append(vec)
		return np.array(result)

	def save(self, filename):
		save_pickle(self.w2vModel, filename)

class TfidfWord2Vec:
	def __init__(self):
		pass

	def fit(self, lines):
		pass 

	def transform(self, lines):
		pass


def vectorize_word2vec(trainfile, testfile):
	train = []
	train_labels = []
	test = []
	test_labels = []
	# Train
	lines = read_lines(trainfile)
	train_lines = split_lines(lines)
	for line in train_lines:
		train.append(line[1:])
		train_labels.append(line[0])
	train_labels = np.array(train_labels)
	# Test
	lines = read_lines(testfile)
	test_lines = split_lines(lines)
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