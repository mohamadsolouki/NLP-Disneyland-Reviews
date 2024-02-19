import pandas as pd
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pyLDAvis.gensim


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

# Print the topics
for topic in lda_model.print_topics():
    print(f"Topic {topic[0]}: {topic[1]}")
    print("\n")

# Save the Word2Vec model
model.wv.save('word2vec.model')

# Load the model
word2vec_model = KeyedVectors.load('word2vec.model')

# Use the Word2Vec model to find similar words
similar_words = word2vec_model.similar_by_word('line')
print(f"The words similar to 'line' are: {similar_words}")
print("\n")

# Use the Word2Vec model to find the similarity between two words
similarity = word2vec_model.similarity('queue', 'long')
print(f"The similarity between 'queue' and 'long' is {similarity}")

# Visualizing using pyLDAvis
import pyLDAvis
import IPython

# Visualize the topics
if IPython.get_ipython() is not None:
    pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_topics.html')

# Visualize the Word2Vec model using t-SNE
# Get the word vectors
word_vectors = word2vec_model.vectors

# Reduce the dimensionality of the word vectors using t-SNE
tsne = TSNE(n_components=2, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot the word vectors in 2D
plt.figure(figsize=(10, 10))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], marker='o')
plt.title('t-SNE visualization of Word2Vec model')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.tight_layout()
plt.savefig('images/word2vec_tsne.png')
plt.show()

