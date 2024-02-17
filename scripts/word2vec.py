import pandas as pd
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

# Load the preprocessed reviews
preprocessed_reviews = []
df = pd.read_csv('data/cleaned_reviews.csv')
for review in df['Clean_Text']:
    preprocessed_reviews.append(word_tokenize(review))

# Train Word2Vec model
model = Word2Vec(preprocessed_reviews, window=5, min_count=1, workers=4)

# Represent reviews as vectors
review_vectors = [model.wv[token] for review in preprocessed_reviews for token in review]

# Apply LDA for topic modeling
dictionary = Dictionary(preprocessed_reviews)
corpus = [dictionary.doc2bow(review) for review in preprocessed_reviews]
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary)

# Print topics in tabular format with headers
topics = lda_model.print_topics(num_words=5)
print(f"The topics are: {topics}")

# Save the Word2Vec model
model.wv.save('word2vec.model')

# Load the model
word2vec_model = KeyedVectors.load('word2vec.model')

# Use the Word2Vec model to find similar words
similar_words = word2vec_model.similar_by_word('line')
print(f"The words similar to 'line' are: {similar_words}")

# Step 15: Use the Word2Vec model to find the similarity between two words
similarity = word2vec_model.similarity('good', 'bad')
print(f"The similarity between 'good' and 'bad' is {similarity}")
