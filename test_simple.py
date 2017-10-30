from models.svm import SVM
import numpy as np

# Data
means = [[2,2], [4,2]]
cov = [[.3, .2], [.2, .3]]
n = 10
X0 = np.random.multivariate_normal(means[0], cov, n)
X1 = np.random.multivariate_normal(means[1], cov, n)
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(n), -1. * np.ones(n)))

# Model
clf = SVM()
# Fitting
clf.fit(X, y)

w, b = clf.get_weights()
print 'w'
print w
print 'b'
print b
# Predict
print 'pred'
print clf.predict(np.array([[1,2],[3,4]]))