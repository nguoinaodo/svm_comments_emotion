import re

POSITIVE = '1'
NEGATIVE = '2'
NEUTRAL = '3'

def main():	
	# Mix
	positive_train, negative_train, neutral_train = classify('../data/cleaned/TRAIN.csv')
	# Test
	positive_test, negative_test, neutral_test = classify('../data/cleaned/TEST.csv')
	# # Positive data
	# lines1 = read('../data/cleaned/POSITIVE_TOKENIZED.csv')
	# for l in lines1:
	# 	l = l.strip()
	# 	match = re.search(r'^\s*$', l, flags=re.MULTILINE)	
	# 	if match == None:	
	# 		positive_train.append(l)
	# # Negative data
	# lines2 = read('../data/cleaned/NEGATIVE_TOKENIZED.csv')
	# for l in lines2:
	# 	l = l.strip()
	# 	match = re.search(r'^\s*$', l, flags=re.MULTILINE)	
	# 	if match == None:	
	# 		negative_train.append(l)
	
	save(positive_train, '../data/classified/positive_train')
	save(negative_train, '../data/classified/negative_train')
	save(neutral_train, '../data/classified/neutral_train')
	save(neutral_test, '../data/classified/neutral_test')
	save(positive_test, '../data/classified/positive_test')
	save(negative_test, '../data/classified/negative_test')

def classify(filename):
	lines = read(filename)
	positive = []
	negative = []
	neutral = []
	for l in lines:
		l = l.strip()
		label = str(l[0])
		print label
		content = l[2:].strip(' ')
		match = re.search(r'^\s*$', content, flags=re.MULTILINE)	
		if match == None:	
			if label == POSITIVE:
				positive.append(content)
			elif label == NEGATIVE:
				negative.append(content)
			else:
				neutral.append(content)
		else:
			print match.group(0)
	return positive, negative, neutral		
			
def save(data, filename):
	with open(filename, 'w') as f:
		for line in data:
			f.write(line + '\n')

def read(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	return lines

main()					