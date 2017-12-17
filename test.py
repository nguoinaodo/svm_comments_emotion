# !/usr/bin/python
# -*- coding: utf8 -*-

import sys
from pyvi.pyvi import ViTokenizer, ViPosTagger
from utils import load_pickle

s1 = ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội")
print(s1)
s2 = ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội"))
print(s2)

if __name__ == '__main__':
	s = u'mua thận giá rẻ'
	s = ViTokenizer.tokenize(unicode(s))
	model = load_pickle('result/tfidf_svm_2/est.pkl')
	print(model.predict([s]))


