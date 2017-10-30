import numpy as np
import re

def read_stop_words(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		stop_words = []
		for line in lines:
			w = '_'.join(line.strip().split())
			stop_words.append(w)
	return stop_words

def eliminate_stop_words(doc, stop_words):
	for w in stop_words:
		reg = re.compile('\s+' + w + '\s+')
		doc = re.sub(reg, ' ', doc)
	return doc	