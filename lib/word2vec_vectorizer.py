from gensim.models import Doc2Vec, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import re
import pickle
from utils import split_lines

"""
Doc2Vec : you can train your dataset using Doc2Vec and then use the sentence vectors.
Average of Word2Vec vectors : You can just take the average of all the word vectors in a sentence. 
		This average vector will represent your sentence vector.
Average of Word2Vec vectors with TF-IDF : this is one of the best approach which I will recommend.
		Just take the word vectors and multiply it with their TF-IDF scores. 
		Just take the average and it will represent your sentence vector.
"""

class Word2VecVectorizer:
	def __init__(self, size=100, window=5, min_count=5, workers=4, word_vectors=None):
		self.size = size
		self.window = window
		self.min_count = min_count
		self.workers = workers
		self.word_vectors = word_vectors

	def fit(self, lines):
		pass

	def transform(self, lines):
		pass

class AverageWord2Vec(Word2VecVectorizer):
	def __init__(self, size=100, window=5, min_count=5, workers=4, word_vectors=None):
		Word2VecVectorizer.__init__(self, size=100, window=5, min_count=5, workers=4, word_vectors=None )

	def fit(self, lines):
		# Split lines to sentences
		sentences = split_lines(lines)
		# Fit model
		w2vModel = Word2Vec(sentences, size=self.size, window=self.window, \
				min_count=self.min_count, workers=self.workers)
		self.word_vectors = w2vModel.wv	

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

class TfidfWord2Vec(Word2VecVectorizer):
	def __init__(self, size=100, window=5, min_count=5, workers=4, tfidf_vectorizer=None, word_vectors=None):
		Word2VecVectorizer.__init__(self, size=100, window=5, min_count=5, workers=4, word_vectors=None)
		self.tfidf_vectorizer = tfidf_vectorizer

	def fit(self, lines):
		# Tfidf
		self.tfidf_vectorizer = TfidfVectorizer()
		self.tfidf_vectorizer.fit(lines)
		# Split lines to sentences  
		sentences = split_lines(lines)
		# Word2vec
		w2vModel = Word2Vec(sentences, size=self.size, window=self.window, \
				min_count=self.min_count, workers=self.workers)
		self.word_vectors = w2vModel.wv
		
	def transform(self, lines):
		result = []
		# Split lines to sentences
		sentences = split_lines(lines)
		tfidf_vocab = self.tfidf_vectorizer.vocabulary_
		tfidf_matrix = self.tfidf_vectorizer.transform(lines)
		for i in range(len(sentences)):
			vec = np.zeros(self.size)
			count = 0
			for token in sentences[i]:
				try:
					vec += self.word_vectors[token] * tfidf_matrix[i, tfidf_vocab[token]]
					count += 1
				except KeyError:
					continue
			if count != 0:
				vec = vec / count
			result.append(vec)
		return np.array(result)	
