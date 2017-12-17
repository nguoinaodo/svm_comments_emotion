from preprocessing.replace import get_replaces, replace
from preprocessing.read import read_lines, split_label_content, split_lines
from preprocessing.vectorize import tfidf_vectorizer
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from preprocess import replace
from utils import save_pickle, load_pickle, make_dir

from pprint import pprint
from time import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve, validation_curve,\
			cross_val_score, GridSearchCV
import matplotlib.pyplot as plt  
import json

#!/usr/bin/python
# -*- coding: utf8 -*-

if __name__ == '__main__':
	############################### Load data
	train2 = load_pickle('../dataset/iphone/tokenized/2-class/train.pkl')	
	test2 = load_pickle('../dataset/iphone/tokenized/2-class/test.pkl')	
	train3 = load_pickle('../dataset/iphone/tokenized/3-class/train.pkl')	
	test3 = load_pickle('../dataset/iphone/tokenized/3-class/test.pkl')	

	############################### Preprocess
	
	# Replace
	replaces = load_pickle('../dataset/iphone/replace/replaces.pkl')	
	
	# train2_data = [replace(line, replaces) for line in train2['data']]
	# test2_data = [replace(line, replaces) for line in test2['data']]
	# train3_data = [replace(line, replaces) for line in train3['data']]
	# test3_data = [replace(line, replaces) for line in test3['data']]

	# save_pickle(train2_data, 'dataset/train2_data')
	# save_pickle(test2_data, 'dataset/test2_data')
	# save_pickle(train3_data, 'dataset/train3_data')
	# save_pickle(test3_data, 'dataset/test3_data')

	train2_data = load_pickle('dataset/train2_data')
	test2_data = load_pickle('dataset/test2_data')
	train3_data = load_pickle('dataset/train3_data')
	test3_data = load_pickle('dataset/test3_data')

	# exit(1)
	############################### Estimator
	estimator = Pipeline([
		('count', CountVectorizer(max_df=.5, min_df=10, max_features=1000, ngram_range=(1, 2))),
		('tfidf', TfidfTransformer()),
		('clf', SVC())
	])
	params = {
		'clf__C': np.logspace(-2, 4, 10),
		'clf__gamma': [.01],
		'clf__kernel': ['rbf', 'sigmoid'] 
	}
	cv = GridSearchCV(estimator, params, cv=3, n_jobs=3, verbose=1, refit=1)
		
	############################### 2 class
	result_path = 'result/tfidf-svm-2'
	make_dir(result_path)	

	# Fit estimator
	cv.fit(train2_data, train2['target'])
	cv_results = cv.cv_results_
	with open('%s/cv_result' % (result_path), 'w') as f:
		f.write('CV_result:\n\n')
		f.write(str(cv_results))
	mean_train_score = cv_results['mean_train_score']
	mean_val_score = cv_results['mean_test_score']

	# Plot
	plt.xlabel('Params')
	plt.ylabel('Score')
	plt.plot(mean_train_score, c='r', label='Training score')
	plt.plot(mean_val_score, c='b', label='Validation score')
	plt.legend()
	plt.savefig('%s/val_curve.png' % result_path)
	plt.show()

	# Best estimator 
	best_estimator = cv.best_estimator_
	save_pickle(best_estimator, '%s/est.pkl' % result_path)
	best_estimator = load_pickle('%s/est.pkl' % result_path)

	# Test
	with open('%s/test-result' % result_path, 'w') as f:
		test_pred = best_estimator.predict(test2_data)
		f.write(classification_report(test2['target'], test_pred))

	############################### 3 class
	result_path = 'result/tfidf-svm-3'
	make_dir(result_path)	

	# Fit estimator
	cv.fit(train3_data, train3['target'])
	cv_results = cv.cv_results_
	with open('%s/cv_result' % (result_path), 'w') as f:
		f.write('CV_result:\n\n')
		f.write(str(cv_results))
	mean_train_score = cv_results['mean_train_score']
	mean_val_score = cv_results['mean_test_score']

	# Plot
	plt.xlabel('Params')
	plt.ylabel('Score')
	plt.plot(mean_train_score, c='r', label='Training score')
	plt.plot(mean_val_score, c='b', label='Validation score')
	plt.legend()
	plt.savefig('%s/val_curve.png' % result_path)
	plt.show()

	# Best estimator 
	best_estimator = cv.best_estimator_
	save_pickle(best_estimator, '%s/est.pkl' % result_path)
	best_estimator = load_pickle('%s/est.pkl' % result_path)
	
	# Test
	with open('%s/test-result' % result_path, 'w') as f:
		test_pred = best_estimator.predict(test3_data)
		f.write(classification_report(test3['target'], test_pred))

	############################### Vocab	
	vocab = best_estimator.get_params()['count'].vocabulary_.keys()
	with open('result/vocab', 'w') as f:
		for w in vocab:
			print(w)
			f.write('%s\n' % w.encode('utf8'))
	