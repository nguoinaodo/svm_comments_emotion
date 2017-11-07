import numpy as np 

# Read lines of file
def read_lines(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	return lines

# Split labels
def split_label_content(lines):
	labels = []
	contents = []
	for line in lines:
		contents.append(line[2:])
		labels.append(line[0])
	return np.array(labels), np.array(contents)

# Split each line into tokens array
def split_lines(lines):
	result = []
	reg = r'\s+|\.+|\,+|\b|&|~|"'
	for line in lines:
		tokens = re.split(reg, line)
		result.append(tokens)
	return result


