from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils import split_lines, save_pickle, load_pickle

class TfidfLSAVectorizer:
	def __init__(self, size=100):
		self.size = 100

	def fit(self, lines):
		self.tfidf_vectorizer = TfidfVectorizer()
		self.lsa = TruncatedSVD(n_components=self.size)
		transformed = self.tfidf_vectorizer.fit_transform(lines)
		self.lsa.fit(transformed)

	def transform(self, lines):
		# Tf-idf transformation
		tfidf_matrix = self.tfidf_vectorizer.transform(lines)
		# LSA (truncated SVD)
		truncated = self.lsa.transform(tfidf_matrix)
		return truncated