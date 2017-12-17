import os
import pickle

# Save pkl file
def save_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

# Load pkl file
def load_pickle(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)