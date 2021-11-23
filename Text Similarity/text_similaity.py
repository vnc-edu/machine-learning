# Source: https://stackoverflow.com/questions/44862712/td-idf-find-cosine-similarity-between-new-document-and-dataset/
# 44863365#44863365
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Generate DF
df = \
    pd.DataFrame({'jobId': [1, 2, 3, 4, 5],
                  'serviceId': [99, 88, 77, 66, 55],
                  'text': ['i like ice cream',
                           'i love ice land',
                           'you hate ice cubes',
                           'we love our mother land',
                           'we conquer foreign land'
                           ]})

# Show DF
df

# Vectorizer to convert a collection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()

# Learn vocabulary and idf, return term-document matrix.
tfidf = vectorizer.fit_transform(df['text'].values.astype('U'))

# Array mapping from feature integer indices to feature name
words = vectorizer.get_feature_names()

# Compute cosine similarity between samples in X and Y.
similarity_matrix = cosine_similarity(tfidf, tfidf)

# Matrix product
print('Similarity Matrix')
print(similarity_matrix)

# Instead of using fit_transform, you need to first fit
# the new document to the TFIDF matrix corpus like this:
queryTFIDF = TfidfVectorizer().fit(words)

# We can check that using a new document text
# query = 'do you like foreign land'
# query = 'we love our mother land'
query = 'Nobody cares for desert'

# Now we can 'transform' this vector into that matrix shape by using the transform function:
queryTFIDF = queryTFIDF.transform([query])

# As we transformed our query in a tfidf object
# we can calculate the cosine similarity in comparison with
# our pevious corpora
cosine_similarities = cosine_similarity(queryTFIDF, tfidf).flatten()

# Get most similar jobs based on next text
related_product_indices = cosine_similarities.argsort()[:-11:-1]
print('related_product_indices')
print(related_product_indices)
# array([3, 2, 1, 4, 0])
