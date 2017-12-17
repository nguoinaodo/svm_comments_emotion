import re
# !/usr/bin/python
# -*- coding: utf8 -*-

def replace(s, replaces):
	for r in replaces:
		s = re.sub(r[0], r[1], s)
	return s	

