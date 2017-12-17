from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
		TfidfTransformer
import numpy as np
from utils import split_lines, save_pickle, load_pickle
from sklearn.base import BaseEstimator, TransformerMixin

class VNEmoLexCountVectorizer(BaseEstimator, TransformerMixin):
	def __init__(self, emodic):
		self.emodic = emodic

	def fit(self, lines):
		self.count_vectorizer = CountVectorizer()
		self.count_vectorizer.fit(lines)

	def transform(self, lines):
		counted = self.count_vectorizer.transform(lines).tolil()
		vocab = self.count_vectorizer.vocabulary_
		for term in vocab.keys():
			if self.emodic.has_key(term):
				for i in range(len(lines)):
					counted[i, vocab[term]] *= self.emodic[term]
		return counted

class VNEmoLexTfidfVectorizer:
	def __init__(self, emodic):
		self.emodic = emodic

	def fit(self, lines):
		self.count_vectorizer = CountVectorizer()
		self.count_vectorizer.fit(lines)
		self.tfidf_transformer = TfidfTransformer()
		counted = self.count_vectorizer.transform(lines).tolil()
		vocab = self.count_vectorizer.vocabulary_
		for term in vocab.keys():
			if self.emodic.has_key(term):
				for i in range(len(lines)):
					counted[i, vocab[term]] *= self.emodic[term]
		self.tfidf_transformer.fit(counted)

	def transform(self, lines):
		counted = self.count_vectorizer.transform(lines).tolil()
		vocab = self.count_vectorizer.vocabulary_
		for term in vocab.keys():
			if self.emodic.has_key(term):
				for i in range(len(lines)):
					counted[i, vocab[term]] *= self.emodic[term]
		tfidf_score = self.tfidf_transformer.transform(counted)
		return tfidf_score
