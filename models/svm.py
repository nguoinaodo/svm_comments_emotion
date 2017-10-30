import numpy as np 
import matplotlib.pyplot as plt 
from cvxopt import matrix, solvers

class SVM:
	def __init__(self):
		self._epsilon = 1e-6

	def set_params(self):
		pass

	def get_params(self):	
		pass	

	def get_weights(self):
		return self._w, self._b

	# Fit data to calculate weights of boundary	
	def fit(self, X, y):
		"""
			X: training features (NxD)
			y: training labels (N), yn in {-1, 1}

			Result: weights: w (D), b (1)
		"""
		# Solve dual problem
		lamda = self._solve_dual(X, y)
		# Get support set: {n: lamda_n != 0}
		S = np.where(lamda > self._epsilon)[0]
		# Calculate weights
		self._w, self._b = self._calculate_weights(X, y, lamda, S)	

	# Solve dual problem
	def _solve_dual(self, X, y):
		N = X.shape[0]
		V = y.reshape(N, 1) * X # NxD
		K = matrix(V.dot(V.T)) # NxN
		p = matrix(-np.ones((N, 1)))

		G = matrix(-np.eye(N))
		h = matrix(np.zeros((N, 1)))
		A = matrix(y.reshape(1, N))
		b = matrix(np.ones((1, 1)))
		solvers.options['show_progress'] = False
		sol = solvers.qp(K, p, G, h, A, b)
		l = np.array(sol['x']).reshape(N)
		return l

	# Calculate weights
	def _calculate_weights(self, X, y, lamda, S):
		lS = lamda[S]
		XS = X[S, :]
		yS = y[S]
		# w
		w = (lS * yS).dot(XS)
		# b
		b = np.mean(yS - w.dot(XS.T))
		return w, b	

	# Predict class labels (1, -1)	
	def predict(self, X):
		N, D = X.shape
		y = self._b + self._w.dot(X.T)
		for i in range(len(y)):
			if y[i] >= 0:
				y[i] = 1
			else:
				y[i] = -1
		return y

	# Accuracy of classification	
	def accuracy(self, X, y):
		pred = self.predict(X)
		N = len(X)
		true_pred = 0
		for i in range(N):
			if y[i] == pred[i]:
				true_pred += 1
		return 1. * true_pred / N		