import numpy as np
import pickle
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
from preprocessing.vectorize import train_features, \
		train_labels, test_features, test_labels
clf = OneVsRestClassifier(DecisionTreeClassifier())
# clf = GaussianNB()
# clf = OneVsRestClassifier(SVC(kernel='linear'), param_grid)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
# 		hidden_layer_sizes=(10, 10), random_state=1) # 200PCA
# Model
# param_grid = {'C': [1e1, 1e2, 1e3, 5e3, 1e4],\
#         'gamma': [0.0001, 0.0005, 0.001, .01, .1]}#, 0.001, 0.005, 0.01, 0.1]}
# clf = GridSearchCV(SVC(kernel='linear'), param_grid)
# clf = SGDClassifier()
# Fit
for n in [500, 600, 700, 800, 900, 1000, 1500, 2000]:
	pca = PCA(n_components=n, svd_solver='full').fit(train_features.toarray())
	with open('pca.pkl', 'wb') as f:
		pickle.dump(pca, f)
	train_features_pca = pca.transform(train_features.toarray())
	test_features_pca = pca.transform(test_features.toarray())
	
	clf.fit(train_features_pca, train_labels)
	# Predict
	print clf.score(test_features_pca, test_labels)
	print np.where(clf.predict(test_features_pca) == 1)



# now you can save it to a file
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

