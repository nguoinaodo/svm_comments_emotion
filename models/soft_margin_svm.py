import numpy as np
from cvxopt import matrix, solvers
from svm import SVM

class SoftMarginSVM(SVM):
	def __init__(self):
		SVM.__init__(self)

		