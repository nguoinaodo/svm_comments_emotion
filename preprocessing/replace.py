import re

def get_replaces(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		replaces = [] 
		for l in lines:
			l = l[:-1].split('&&')
			reg = re.compile(l[0], flags=re.I)
			r = [reg, l[1]]
			replaces.append(r)
	return replaces

def replace(fromfile, tofile, replaces):
	with open(fromfile, 'r') as f:
		s = f.read()
		for r in replaces:
			s = re.sub(r[0], r[1], s)
		with open(tofile, 'w') as f1:
			f1.write(s)
			
def main():
	replaces = get_replaces('replace')
	# replace('../data/tokenized/iphone_train_tokenized.csv', \
	# 		'../data/cleaned/iphone_train_tokenized.csv', replaces)
	# replace('../data/tokenized/iphone_dev_tokenized.csv', \
	# 		'../data/cleaned/iphone_dev_tokenized.csv', replaces)
	# replace('../data/tokenized/NEGATIVE_TOKENIZED.csv', \
	# 		'../data/cleaned/NEGATIVE_TOKENIZED.csv', replaces)
	# replace('../data/tokenized/POSITIVE_TOKENIZED.csv', \
	# 		'../data/cleaned/POSITIVE_TOKENIZED.csv', replaces)
	replace('../data/tokenized/TRAIN.csv', \
			'../data/cleaned/TRAIN.csv', replaces)
	replace('../data/tokenized/TEST.csv', \
			'../data/cleaned/TEST.csv', replaces)


main()
