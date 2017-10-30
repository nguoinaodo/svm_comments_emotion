from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import coo_matrix, vstack
import pickle
from dictionary import load_dict


def vectorize_new():
	vectorizer = TfidfVectorizer(token_pattern=r'\w+')
	train = read('DataSMCC/TRAIN.csv')
	test = read('DataSMCC/TEST.csv')
	train_features = []
	train_labels = []
	test_features = []
	test_labels = []
	for x in train:
		train_labels.append(x[0])
		train_features.append(x[2:])
	for x in test:
		test_labels.append(x[0])
		test_features.append(x[2:])
	train_labels = np.array(train_labels).astype(np.int)
	test_labels = np.array(test_labels).astype(np.int)
	train_features = vectorizer.fit_transform(train_features)
	test_features = vectorizer.transform(test_features)
	return train_features, train_labels, test_features, test_labels

def vectorize():
	# Dictionary
	dic, inv_dic = load_dict('data/dictionary/dictionary')
	# Vectorizer
	vectorizer = TfidfVectorizer(vocabulary=dic, token_pattern=r'\w+')
	# Read
	positive_train = read('data/classified/positive_train')
	pos_train_labels = np.ones(len(positive_train))
	negative_train = read('data/classified/negative_train')
	neg_train_labels = 2 * np.ones(len(negative_train))
	# neutral_train = read('data/classified/neutral_train')
	# neu_train_labels = 3 * np.ones(len(neutral_train))
	positive_test = read('data/classified/positive_test')
	pos_test_labels = np.ones(len(positive_test))
	negative_test = read('data/classified/negative_test')
	neg_test_labels = 2 * np.ones(len(negative_test))
	# neutral_test = read('data/classified/neutral_test')
	# neu_test_labels = 3 * np.ones(len(neutral_test))
	# Train
	train_features = np.hstack((positive_train, negative_train))#, neutral_train))
	train_features = vectorizer.fit_transform(train_features)
	train_labels = np.concatenate((pos_train_labels, neg_train_labels))#, neu_train_labels))
	# Test
	test_features = np.hstack((positive_test, negative_test))#, neutral_test))
	test_features = vectorizer.transform(test_features)
	test_labels = np.concatenate((pos_test_labels, neg_test_labels))#, neu_test_labels))

	# now you can save it to a file
	with open('vectorizer.pkl', 'wb') as f:
	    pickle.dump(vectorizer, f)

	return train_features, train_labels, test_features, test_labels

def read(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	return lines

train_features, train_labels, test_features, test_labels = vectorize()
