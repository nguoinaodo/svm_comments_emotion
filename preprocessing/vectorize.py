from sklearn.feature_extraction.text import TfidfVectorizer
from lib.word2vec_vectorizer import AverageWord2Vec
import numpy as np
import re
from scipy.sparse import coo_matrix, vstack
import pickle
from gensim.models import Doc2Vec, Word2Vec
from read import read_lines, split_lines

# Tf-idf vectorizer fit with train contents
def tfidf_vectorizer(train_contents):
	vectorizer = TfidfVectorizer()
	vectorizer.fit(train_contents)
	return vectorizer

# Word2vec vectorizer
def average_word2vec_vectorizer(train_contents, size=100):
	vectorizer = AverageWord2Vec(size=100)
	vectorizer.fit(train_contents)
	return vectorizer