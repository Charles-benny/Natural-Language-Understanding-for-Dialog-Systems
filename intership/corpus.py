from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the corpus and transform the corpus into TF-IDF vectors
X = vectorizer.fit_transform(corpus)

# Print the feature names (words)
X = vectorizer.fit_transform(corpus)

# Print the TF-IDF matrix
print("TF-IDF matrix:")
print(X.toarray())
vectorizer.get_feature_names()