from sklearn.feature_extraction.text import TfidfVectorizer
from lib.word2vec_vectorizer import AverageWord2Vec, TfidfWord2Vec
from lib.lsa_vectorizer import TfidfLSAVectorizer
from lib.vnemolex_vectorizer import VNEmoLexCountVectorizer, VNEmoLexTfidfVectorizer
import numpy as np
import re
from scipy.sparse import coo_matrix, vstack
import pickle
from gensim.models import Doc2Vec, Word2Vec
from read import read_lines, split_lines
from dictionary import read_vnemolex

# Tf-idf vectorizer fit with train contents
def tfidf_vectorizer(train_contents):
	vectorizer = TfidfVectorizer()
	vectorizer.fit(train_contents)
	return vectorizer

# Word2vec vectorizer
def average_word2vec_vectorizer(train_contents, size=1000):
	vectorizer = AverageWord2Vec(size=size)
	vectorizer.fit(train_contents)
	return vectorizer

# Tf-idf word2vec vectorizer
def tfidf_word2vec_vectorizer(train_contents, size=1000):
	vectorizer = TfidfWord2Vec(size=size)
	vectorizer.fit(train_contents)
	return vectorizer

# Tf-idf LSA vectorizer
def tfidf_lsa_vectorizer(train_contents, size=1000):
	vectorizer = TfidfLSAVectorizer(size=size)
	vectorizer.fit(train_contents)
	return vectorizer

# Count vector add by emotion point
def count_emotion_vectorizer(train_contents, dicfile):
	dic = read_vnemolex(dicfile)
	vectorizer = VNEmoLexCountVectorizer(emodic=dic)
	vectorizer.fit(train_contents)
	return vectorizer

# Count vector add by emotion point, then transform by tfidf
def tfidf_emotion_vectorizer(train_contents, dicfile):
	dic = read_vnemolex(dicfile)
	vectorizer = VNEmoLexTfidfVectorizer(emodic=dic)
	vectorizer.fit(train_contents)
	return vectorizer