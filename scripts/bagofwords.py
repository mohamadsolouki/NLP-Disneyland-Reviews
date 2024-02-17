# bag_of_words.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# Load the preprocessed data
file_path = 'data/cleaned_reviews.csv'
clean_df = pd.read_csv(file_path)

# Initialize CountVectorizer and fit and transform
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 3), min_df=5, max_df=0.5)
count_vectors = count_vectorizer.fit_transform(clean_df['Clean_Text'])


# Function to display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic {}:".format(topic_idx + 1))
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


# Function to plot word frequencies
def plot_word_frequencies(words, freqs, title):
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(words)), freqs, align='center')
    plt.yticks(range(len(words)), words)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
    plt.xlabel('Frequency')
    plt.title(title)
    plt.show()


# Function to generate a word cloud
def generate_word_cloud(topic, feature_names, no_top_words):
    word_freqs = {feature_names[i]: topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# Sum up the counts of each vocabulary word
sum_words = count_vectors.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]
sorted_words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Display the top N most frequent words
top_n = 30
print("\nTop {} most frequent words/ngrams:".format(top_n))
print("-" * 40)
for word, freq in sorted_words_freq[:top_n]:
    print("{:<20} : {}".format(word, freq))
print("-" * 40)

# Plotting the top N words/ngrams
top_words, top_freqs = zip(*sorted_words_freq[:top_n])
plot_word_frequencies(top_words, top_freqs, 'Top {} Words/N-grams Frequency'.format(top_n))

# Number of topics and top words to display
n_topics = 4
no_top_words = 10

# Initialize and fit LDA, LSA, and NMF models
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(count_vectors)
lsa = TruncatedSVD(n_components=n_topics).fit(count_vectors)
nmf = NMF(n_components=n_topics, random_state=42).fit(count_vectors)

# Display topics for each model
print("\nLDA Model Topics:")
display_topics(lda, count_vectorizer.get_feature_names_out(), no_top_words)
print("\nLSA Model Topics:")
display_topics(lsa, count_vectorizer.get_feature_names_out(), no_top_words)
print("\nNMF Model Topics:")
display_topics(nmf, count_vectorizer.get_feature_names_out(), no_top_words)

# Generate word clouds for LDA topics
for topic_idx, topic in enumerate(lda.components_):
    print("Word Cloud for LDA Topic {}:".format(topic_idx + 1))
    generate_word_cloud(topic, count_vectorizer.get_feature_names_out(), no_top_words)


# t-SNE Visualization for LDA
def tsne_visualization(model, data):
    print("\nPerforming t-SNE Visualization...")
    topic_weights = model.transform(data)
    tsne_model = TSNE(n_components=2, verbose=0, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(topic_weights)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 5))
    plt.scatter(tsne_lda[:, 0], tsne_lda[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization of LDA Topics')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    return tsne_lda


if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass
