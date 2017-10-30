import pickle
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
# and later you can load it
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
	vectorizer = pickle.load(f)
with open('test_comments', 'r') as f:
	sentences = f.readlines()
with open('pca.pkl', 'rb') as f:
	pca = pickle.load(f)
 
features = vectorizer.transform(sentences).toarray()
features = pca.transform(features)
print clf.predict(features)
