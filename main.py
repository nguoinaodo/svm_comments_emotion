from preprocessing.replace import get_replaces, replace
from preprocessing.read import read_lines, split_label_content, split_lines
from preprocessing.vectorize import tfidf_vectorizer, average_word2vec_vectorizer, \
		tfidf_word2vec_vectorizer
import numpy as np
import pickle
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import os

data_dir = '../dataset/iphone/'

def main(replace=True, preprocess=True, train=True, test=True):
	# Params
	replace_file = '%sreplace/replace.txt' % data_dir
	tokenized_dir = '%stokenized/' % data_dir
	cleaned_dir = '%scleaned/' % data_dir

	vectorizer_type = 'avg-w2v'
	vector_size = 100 # only use if vectorizer type is w2v, else None

	Cs = [1, 5, 7, 10]
	gammas = [1]
	kernels= ['rbf', 'sigmoid']

	vectorized_dir = '%svectorized/%s/' % (data_dir, vectorizer_type)
	if vector_size:
		vectorized_dir += '%s/' % vector_size

	features_file = '%sfeatures.pkl' % vectorized_dir
	labels_file = '%slabels.pkl' % vectorized_dir

	do_save_vectorizer = False
	do_load_vectorizer = False

	make_dirs(cleaned_dir)
	make_dirs(vectorized_dir)
	# Test models
	for C in Cs:
		for gamma in gammas:
			for kernel in kernels:
				result_dir = 'result/C%d-gamma%.2f-kernel_%s-%s/' % (C, gamma, kernel, vectorizer_type) 
				if vector_size:
					result_dir = '%s%d/' % (result_dir, vector_size)
				vectorizer_file = get_vectorizer_file(result_dir)
				model_file = get_model_file(result_dir)
				score_file = get_score_file(result_dir)
				make_dirs(result_dir)
				# Replace
				if replace:
					_replace('%sTRAIN.csv' % tokenized_dir, '%sTRAIN.csv' % cleaned_dir)
					_replace('%sTEST.csv' % tokenized_dir, '%sTEST.csv' % cleaned_dir)
				# Read data then vectorize
				if preprocess:
					train_labels, train_contents =  _read('%sTRAIN.csv' % cleaned_dir)
					test_labels, test_contents =  _read('%sTEST.csv' % cleaned_dir)
					# Vectorize
					train_features, test_features = _vectorize(train_contents, \
							test_contents, vectorizer_file=vectorizer_file, vectorizer_type=vectorizer_type, \
							save=do_save_vectorizer, load=do_load_vectorizer, vector_size=vector_size)
					# Save features and labels
					save_pickle((train_features, test_features), features_file)
					save_pickle((train_labels, test_labels), labels_file)
				else:
					# Load features and labels
					train_features, test_features = load_pickle(features_file)	
					train_labels, test_labels = load_pickle(labels_file)	
				# Train
				if train:
					model = SVC(C=C, gamma=gamma, kernel=kernel)
					model.fit(train_features, train_labels)
					save_pickle(model, model_file)
				else:
					# Load trained model
					model = load_pickle(model_file)
				# Test
				if test:
					score = model.score(test_features, test_labels)
					save_score(score, score_file)
					with open('log/result.txt', 'a') as log:
						log.write('%s_____________%f\n' % (result_dir, score))				
	
# Save pkl file
def save_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

# Load pkl file
def load_pickle(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj

# Model file
def get_model_file(result_dir):
	return '%smodel.pkl' % result_dir

# Score file
def get_score_file(result_dir):
	return '%sscore.txt' % result_dir

# Vectorizer file
def get_vectorizer_file(result_dir):
	return '%svectorizer.pkl' % result_dir

# Make dirs if not exists
def make_dirs(dirpath):
	if os.path.exists(dirpath) == False:
		os.makedirs(dirpath)

# Save test score
def save_score(score, score_file):
	with open(score_file, 'w') as f:
		f.write(str(score))

# Read and split data
def _read(cleaned_file):
	# Raw lines
	lines = read_lines(cleaned_file)
	# Split labels contents
	labels, contents = split_label_content(lines)
	return labels, contents

# Replace
def _replace(raw_file, cleaned_file, replace_file):
	# Replace tokens
	replaces = get_replaces(replace_file)
	# Save replaced file
	replace(raw_file, cleaned_file, replaces)

# Vectorize
def _vectorize(train_contents, test_contents, vectorizer_file=None, \
		vectorizer_type='tf-idf', save=False, load=False, vector_size=100):
	# Vectorizer
	if vectorizer_file and load:
		vectorizer = load_pickle(vectorizer_file)
	else:
		if vectorizer_type == 'tf-idf':
			vectorizer = tfidf_vectorizer(train_contents)
		elif vectorizer_type == 'avg-w2v':
			vectorizer = average_word2vec_vectorizer(train_contents, size=vector_size)
		elif vectorizer_type == 'tfidf-w2v':
			vectorizer = tfidf_word2vec_vectorizer(train_contents, size=vector_size)
		else:
			return	
			
	# Transform
	train_features = vectorizer.transform(train_contents)
	test_features = vectorizer.transform(test_contents)
	# Save
	if save and vectorizer_file:
		save_pickle(vectorizer, vectorizer_file)
	return train_features, test_features

main(replace=False, preprocess=True, train=True, test=True)
