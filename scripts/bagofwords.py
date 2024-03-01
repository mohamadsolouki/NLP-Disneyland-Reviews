import pandas as pd
from gensim.models import LdaModel, Nmf
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
import numpy as np
from nltk import bigrams
import warnings

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
bigram_docs = []
for doc in tokenized_docs:
    grams = list(bigrams(doc))
    bigram_docs.append(['_'.join(gram) for gram in grams])

# Create a Gensim dictionary from the tokenized docs
gensim_dictionary = Dictionary(bigram_docs)

# Filter extremes to mirror CountVectorizer's min_df and max_df
gensim_dictionary.filter_extremes(no_below=10, no_above=0.3)

# Convert the dictionary to a bag of words corpus for reference
gensim_corpus = [gensim_dictionary.doc2bow(doc) for doc in bigram_docs]

def compute_coherence(model, gensim_corpus, texts, coherence_measure='c_v'):
    coherence_model = CoherenceModel(model=model, texts=texts, corpus=gensim_corpus, dictionary=model.id2word, coherence=coherence_measure)
    return coherence_model.get_coherence()

# Set parameters and run the models
num_topics = range(3, 11) # Adjust the number of topics as needed
coherence_scores = {'LDA': [], 'NMF': []}
models = {'LDA': [], 'NMF': []}

for k in num_topics:
    lda_model = LdaModel(corpus=gensim_corpus, num_topics=k, id2word=gensim_dictionary, passes=10, random_state=42)
    models['LDA'].append(lda_model)
    coherence_scores['LDA'].append(compute_coherence(lda_model, gensim_corpus, bigram_docs))

    nmf_model = Nmf(corpus=gensim_corpus, num_topics=k, id2word=gensim_dictionary, passes=10, random_state=42)
    models['NMF'].append(nmf_model)
    coherence_scores['NMF'].append(compute_coherence(nmf_model, gensim_corpus, bigram_docs))

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
    print("Topic %d:" % (topic_idx), " ".join([word for word,_ in word_probs]))

print("\nBest NMF Model Topics:")
for topic_idx, topic in enumerate(best_nmf_model.show_topics(num_topics=best_nmf_model.num_topics, formatted=False)):
    word_probs = topic[1]
    print("Topic %d:" % (topic_idx), " ".join([word for word, _ in word_probs]))

# Save models
best_lda_model.save('models/bow_lda_model')
best_nmf_model.save('models/bow_nmf_model')

if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass
