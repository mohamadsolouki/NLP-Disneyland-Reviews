import pandas as pd
from gensim.models import Word2Vec, Phrases
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.cluster import KElbowVisualizer

# Suppress warnings that do not affect the analysis
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the cleaned reviews
df = pd.read_csv('data/cleaned_reviews.csv')
preprocessed_docs = df['Clean_Text'].tolist()

# Tokenized docs
tokenized_docs = [doc.split() for doc in preprocessed_docs]

# Generate bigrams and trigrams using Gensim's Phrases
bigram = Phrases(tokenized_docs, min_count=10)
trigram = Phrases(bigram[tokenized_docs])
bigram_trigram_docs = [trigram[bigram[doc]] for doc in tokenized_docs]

# Train a Word2Vec model
model = Word2Vec(bigram_trigram_docs, vector_size=100, workers=4)

# Get the vectors for all words/phrases in the model
vectors = model.wv.vectors

# Standardize the vectors using StandardScaler
scaler = StandardScaler()
vectors_standardized = scaler.fit_transform(vectors)

# Fit KMeans models and calculate distortion for a range of cluster numbers using parallel processing
def calculate_distortion(k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vectors_standardized)
    return k, kmeans.inertia_

num_clusters = range(2, 31)
distortion_scores = dict(Parallel(n_jobs=-1)(delayed(calculate_distortion)(k) for k in num_clusters))

# Plot distortion scores
plt.figure(figsize=(12, 6))
plt.plot(list(distortion_scores.keys()), list(distortion_scores.values()), marker='o')
plt.title('Distortion Score as a Function of the Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion Score')
plt.show()

# Find the best number of clusters using the elbow method
best_k = min(distortion_scores, key=distortion_scores.get)
print(f'Best number of clusters: {best_k}')

# Train a KMeans model with the best number of clusters
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
best_kmeans.fit(vectors_standardized)

# Reduce the dimensionality of the cluster centers for visualization
pca = PCA(n_components=2)
cluster_centers_2d = pca.fit_transform(best_kmeans.cluster_centers_)

# Visualize the cluster centers
plt.figure(figsize=(8, 6))
plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1])
for i, center in enumerate(cluster_centers_2d):
    plt.annotate(str(i), (center[0], center[1]))
plt.title('Cluster Centers')
plt.show()

# Assign each word to a cluster
word_to_cluster = {word: best_kmeans.labels_[i] for i, word in enumerate(model.wv.index_to_key)}

# Print the top words for each cluster
for cluster in range(best_k):
    print(f'Cluster {cluster}:')
    top_words = [word for word, clust in word_to_cluster.items() if clust == cluster]
    print(top_words[:10])  # Print```python


# Load dataset
df = pd.read_csv('data/cleaned_reviews.csv')
docs = df['Clean_Text'].values

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(docs)
tfidf_norm = StandardScaler().fit_transform(tfidf.toarray())

# Optimal cluster finding
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,30), timings=False)
visualizer.fit(tfidf_norm)
visualizer.show()

best_k = visualizer.elbow_value_
print(f"Best number of clusters: {best_k}")

# Clustering
km = KMeans(n_clusters=7)
clusters = km.fit_predict(tfidf_norm)

# PCA for dimensionality reduction for visualization
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(tfidf_norm)
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))

for i in range(best_k):
    points = np.array([scatter_plot_points[j] for j in range(len(scatter_plot_points)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=25, c=colors[i], label=f'Cluster {i}')

ax.legend()
plt.title('Clusters by PCA Components')
plt.show()

# Print top terms per cluster
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()
for i in range(best_k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {', '.join(top_terms)}")

if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass