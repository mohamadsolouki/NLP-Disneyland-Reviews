import pandas as pd
import gensim
from gensim.models import LdaModel, Nmf
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
import numpy as np
from nltk import bigrams, trigrams
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pyLDAvis.gensim

# Suppress warnings that do not affect the analysis
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the cleaned reviews
df = pd.read_csv('data/cleaned_reviews.csv')

# Loading list of strings containing the preprocessed documents
# df should be defined previously and should contain a 'Clean_Text' column
preprocessed_docs = df['Clean_Text'].tolist()

# Tokenized docs needed for coherence score calculation
tokenized_docs = [doc.split() for doc in preprocessed_docs]

# Generate bigrams and trigrams
bigram_trigram_docs = []
for doc in tokenized_docs:
    grams = list(bigrams(doc)) + list(trigrams(doc))
    bigram_trigram_docs.append(['_'.join(gram) for gram in grams])

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Convert the documents to a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in bigram_trigram_docs])

# Convert the TF-IDF matrix to a Gensim corpus
gensim_corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix, documents_columns=False)

# Create a Gensim dictionary from the bigram_trigram_docs
gensim_dictionary = Dictionary([' '.join(doc).split() for doc in bigram_trigram_docs])

# Filter extremes 
gensim_dictionary.filter_extremes(no_below=5, no_above=0.5)

# Convert the documents to a Gensim corpus
gensim_corpus = [gensim_dictionary.doc2bow(doc) for doc in [' '.join(doc).split() for doc in bigram_trigram_docs]]


def compute_coherence(model, gensim_corpus, texts, coherence_measure='c_v'):
    coherence_model = CoherenceModel(model=model, texts=texts, corpus=gensim_corpus, dictionary=model.id2word, coherence=coherence_measure)
    return coherence_model.get_coherence()

# Set parameters and run the models
num_topics = range(3, 11)  # Adjust the number of topics as needed
coherence_scores = {'LDA': [], 'NMF': []}
models = {'LDA': [], 'NMF': []}


# Compute coherence scores for different number of topics. pay attention toIndexError: index 942585 is out of bounds for axis 1 with size 8541
for k in num_topics:
    lda_model = LdaModel(corpus=gensim_corpus, num_topics=k, id2word=gensim_dictionary, passes=10, random_state=42)
    models['LDA'].append(lda_model)
    coherence_scores['LDA'].append(compute_coherence(lda_model, gensim_corpus, bigram_trigram_docs))

    nmf_model = Nmf(corpus=gensim_corpus, num_topics=k, id2word=gensim_dictionary, passes=10, random_state=42)
    models['NMF'].append(nmf_model)
    coherence_scores['NMF'].append(compute_coherence(nmf_model, gensim_corpus, bigram_trigram_docs))

    print(f'Num Topics: {k}, LDA Coherence: {coherence_scores["LDA"][-1]}, NMF Coherence: {coherence_scores["NMF"][-1]}')


# Plot coherence scores
plt.figure(figsize=(12, 6))
plt.plot(num_topics, coherence_scores['LDA'], label='LDA')
plt.plot(num_topics, coherence_scores['NMF'], label='NMF')
plt.title('Coherence Score as a Function of the Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.legend(title='Model', loc='best')
plt.show()

# Find the model with the highest coherence and print the topics
best_lda_index = np.argmax(coherence_scores['LDA'])
best_nmf_index = np.argmax(coherence_scores['NMF'])

best_lda_model = models['LDA'][best_lda_index]
best_nmf_model = models['NMF'][best_nmf_index]

# Print the topics from the best models
print("\nBest LDA Model Topics:")
for topic_idx, topic in enumerate(best_lda_model.show_topics(num_topics=best_lda_model.num_topics, formatted=False)):
    word_probs = topic[1]
    print("Topic %d:" % (topic_idx), " ".join([word for word, _ in word_probs]))

print("\nBest NMF Model Topics:")
for topic_idx, topic in enumerate(best_nmf_model.show_topics(num_topics=best_nmf_model.num_topics, formatted=False)):
    word_probs = topic[1]
    print("Topic %d:" % (topic_idx), " ".join([word for word, _ in word_probs]))

# Save models using pickle module
with open('models/tfidf_lda_model', 'wb') as f:
    pickle.dump(best_lda_model, f)

with open('models/tfidf_nmf_model', 'wb') as f:
    pickle.dump(best_nmf_model, f)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(best_lda_model, gensim_corpus, gensim_dictionary)
vis


if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass
