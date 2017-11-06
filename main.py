import numpy as np
import pickle
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

# Data
from preprocessing.vectorize import vectorize_word2vec
train_features, train_labels, test_features, test_labels = vectorize_word2vec(
		'data/cleaned/TRAIN.csv', 'data/cleaned/TEST.csv')
# Model
# clf = DecisionTreeClassifier()
# clf = GaussianNB()

clf = SVC(kernel='linear', C=1)

# clf = MLPClassifier(hidden_layer_sizes=(10,10,10),\
# 		max_iter=1000) # 200PCA
# param_grid = {'C': [1e1, 1e2, 1e3, 5e3, 1e4],\
#         'gamma': [0.0001, 0.0005, 0.001, .01, .1]}#, 0.001, 0.005, 0.01, 0.1]}
# clf = GridSearchCV(SVC(kernel='linear'), param_grid)
# clf = SGDClassifier()

# Fit and log
logdir = 'log/logw2v/'
pcadir = 'log/pca/'
for c in [1]:

	# clf = SVC(kernel='linear', C=c)

	for n in [120]:
		# Log file
		filename = str(time.time())
		with open(logdir + filename, 'w') as f:
			f.write('%s\n' % clf)

			# PCA
			# pca = PCA(n_components=n, svd_solver='full')\
			# # with open(pcadir + filename + '.pkl', 'wb') as f1:
			# # 	pickle.dump(pca, f1)
			# pca.fit(train_features)
			# f.write('%s\n' % pca)
			# train_features_pca = pca.transform(train_features)
			# test_features_pca = pca.transform(test_features)
			
			# Fit
			# clf.fit(train_features_pca, train_labels)

			clf.fit(train_features, train_labels)

			# Predict
			# pred = clf.predict(test_features_pca)

			pred = clf.predict(test_features)
			f.write('Predict:\n')
			f.write('%s\n' % pred)

			# score = clf.score(test_features_pca, test_labels)

			score = clf.score(test_features, test_labels)
			print 'Score: %f' % score
			f.write('Score: %f\n' % score)
			with open('score', 'a') as f2:
				f2.write('%s: %f\n' % (filename, score))

# now you can save it to a file
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

