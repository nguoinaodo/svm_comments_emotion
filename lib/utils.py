# Split each line into tokens array
def split_lines(lines):
	result = []
	reg = r'\s+|\.+|\,+|\b|&|~|"'
	for line in lines:
		tokens = re.split(reg, line)
		result.append(tokens)
	return result

# Save pkl file
def save_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

# Load pkl file
def load_pickle(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj
