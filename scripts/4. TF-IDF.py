import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# Load the preprocessed data
file_path = 'data/cleaned_reviews.csv'
clean_df = pd.read_csv(file_path)

# Initialize TfidfVectorizer and fit and transform
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5, max_df=0.5)
tfidf_vectors = tfidf_vectorizer.fit_transform(clean_df['Clean_Text'])

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
    plt.savefig('images/word_frequencies.png')
    plt.show()

# Function to generate a word cloud
def generate_word_cloud(topic, feature_names, no_top_words):
    word_freqs = {feature_names[i]: topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('images/tfidf_word_cloud.png')
    plt.show()

# Sum up the TF-IDF scores of each vocabulary word
sum_tfidf = tfidf_vectors.sum(axis=0)
words_tfidf = [(word, sum_tfidf[0, idx]) for word, idx in tfidf_vectorizer.vocabulary_.items()]
sorted_words_tfidf = sorted(words_tfidf, key=lambda x: x[1], reverse=True)

# Display the top N words with the highest TF-IDF score
top_n = 30
print("\nTop {} words with the highest TF-IDF scores:".format(top_n))
print("-" * 40)
for word, score in sorted_words_tfidf[:top_n]:
    print("{:<20} : {}".format(word, score))
print("-" * 40)

# Plotting the top N words with the highest TF-IDF scores
top_words, top_scores = zip(*sorted_words_tfidf[:top_n])
plot_word_frequencies(top_words, top_scores, 'Top {} Words with Highest TF-IDF Scores'.format(top_n))

# Number of topics and top words to display
n_topics = 4
no_top_words = 10

# Initialize and fit LDA, LSA, and NMF models
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(tfidf_vectors)
lsa = TruncatedSVD(n_components=n_topics).fit(tfidf_vectors)
nmf = NMF(n_components=n_topics, random_state=42).fit(tfidf_vectors)

# Display topics for each model
print("\nLDA Model Topics:")
display_topics(lda, tfidf_vectorizer.get_feature_names_out(), no_top_words)
print("\nLSA Model Topics:")
display_topics(lsa, tfidf_vectorizer.get_feature_names_out(), no_top_words)
print("\nNMF Model Topics:")
display_topics(nmf, tfidf_vectorizer.get_feature_names_out(), no_top_words)

# Generate word clouds for LDA topics
for topic_idx, topic in enumerate(lda.components_):
    print("Word Cloud for LDA Topic {}:".format(topic_idx + 1))
    generate_word_cloud(topic, tfidf_vectorizer.get_feature_names_out(), no_top_words)

# t-SNE Visualization for LDA
def tsne_visualization(model, data):
    print("\nPerforming t-SNE Visualization...")
    topic_weights = model.transform(data)
    tsne_model = TSNE(n_components=2, verbose=0, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(topic_weights)
    return tsne_lda

# Plot t-SNE
def plot_tsne(tsne_results, title):
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title(title)
    plt.savefig('images/tfidf_tsne_lda.png')
    plt.show()

tsne_lda = tsne_visualization(lda, tfidf_vectors)
plot_tsne(tsne_lda, 't-SNE Visualization of LDA Topics')


if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass