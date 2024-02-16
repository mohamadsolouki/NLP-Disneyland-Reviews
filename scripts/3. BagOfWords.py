from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the preprocessed data
file_path = 'data/cleaned_reviews.csv'
clean_df = pd.read_csv(file_path)

# Initialize CountVectorizer and fit and transform
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 3), min_df=5, max_df=0.5)
count_vectors = count_vectorizer.fit_transform(clean_df['Clean_Text'])

# Sum up the counts of each vocabulary word
sum_words = count_vectors.sum(axis=0)

# Connect the sum counts with the vocabulary words
words_freq = [(word, sum_words[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]

# Sort the words by frequency
sorted_words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Display the top N most frequent words
top_n = 30
print("\nTop {} most frequent words/ngrams:".format(top_n))
print("-" * 40)
for word, freq in sorted_words_freq[:top_n]:
    print("{:<20} : {}".format(word, freq))
print("-" * 40)

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic %d:" % (topic_idx + 1))
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Number of top words to display for each topic
no_top_words = 10

# LDA Model
n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(count_vectors)

print("\nLDA Model Topics:")
display_topics(lda, count_vectorizer.get_feature_names_out(), no_top_words)

# LSA Model
lsa = TruncatedSVD(n_components=n_topics)
lsa.fit(count_vectors)

print("\nLSA Model Topics:")
display_topics(lsa, count_vectorizer.get_feature_names_out(), no_top_words)

# NMF Model
nmf = NMF(n_components=n_topics, random_state=42)
nmf.fit(count_vectors)

print("\nNMF Model Topics:")
display_topics(nmf, count_vectorizer.get_feature_names_out(), no_top_words)

# Extract the top N words/n-grams and their frequencies
top_words, top_freqs = zip(*sorted_words_freq[:top_n])

# Plotting the top N words/n-grams
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_words)), top_freqs, align='center')
plt.yticks(range(len(top_words)), top_words)
plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
plt.xlabel('Frequency')
plt.title('Top {} Words/N-grams Frequency'.format(top_n))
plt.show()

# Assign a topic to each document
doc_topic_dist = lda.transform(count_vectors)

# Display the topic distribution for the first document
print("Document Topic Distribution for the first document:")
print(doc_topic_dist[0])

# For a more global view, you could average the topic distribution across all documents
avg_topic_dist = doc_topic_dist.mean(axis=0)
print("\nAverage Topic Distribution:")
print(avg_topic_dist)

# Plotting average topic distribution
plt.figure(figsize=(8, 6))
plt.bar(range(len(avg_topic_dist)), avg_topic_dist, align='center')
plt.xticks(range(len(avg_topic_dist)), ['Topic {}'.format(i) for i in range(len(avg_topic_dist))])
plt.xlabel('Topics')
plt.ylabel('Average Proportion')
plt.title('Average Topic Distribution Across All Documents')
plt.show()

# Assign a topic to each document
doc_topic_dist = lda.transform(count_vectors)

# Display the topic distribution for the first document
print("Document Topic Distribution for the first document:")
print(doc_topic_dist[0])

# For a more global view, you could average the topic distribution across all documents
avg_topic_dist = doc_topic_dist.mean(axis=0)
print("\nAverage Topic Distribution:")
print(avg_topic_dist)

# Plotting average topic distribution
plt.figure(figsize=(8, 6))
plt.bar(range(len(avg_topic_dist)), avg_topic_dist, align='center')
plt.xticks(range(len(avg_topic_dist)), ['Topic {}'.format(i) for i in range(len(avg_topic_dist))])
plt.xlabel('Topics')
plt.ylabel('Average Proportion')
plt.title('Average Topic Distribution Across All Documents')
plt.show()

from wordcloud import WordCloud

# Generate word clouds for each topic
for topic_idx, topic in enumerate(lda.components_):
    print("Word Cloud for Topic {}: ".format(topic_idx + 1))

    # Create a dictionary with word and its weight
    topic_words_dict = {count_vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]}
    
    # Create a word cloud for the topic
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words_dict)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Qualitative Comparison: Display the top words for each topic
def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Quantitative Comparison: Calculate model perplexity and log-likelihood
def model_scores(model, data):
    print("Perplexity: ", model.perplexity(data))
    print("Log Likelihood: ", model.score(data))

# Visual Analysis: t-SNE Visualization
def tsne_visualization(model, data):
    topic_weights = model.transform(data)
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(topic_weights)
    return tsne_lda

# Statistical Analysis: Held-out Likelihood (if you have a test set `test_data`)
def held_out_likelihood(model, test_data):
    return model.score(test_data)

# Compare the topics
print("LDA Topics:")
display_topics(lda, count_vectorizer.get_feature_names_out())
print("\nLSA Topics:")
display_topics(lsa, count_vectorizer.get_feature_names_out())
print("\nNMF Topics:")
display_topics(nmf, count_vectorizer.get_feature_names_out())

# Model scores for LDA (LSA and NMF do not have perplexity or log-likelihood scores)
print("\nLDA Model Scores:")
model_scores(lda, count_vectors)

# t-SNE Visualization for LDA
tsne_lda = tsne_visualization(lda, count_vectors)
# Now plot the tsne_lda with appropriate labels

