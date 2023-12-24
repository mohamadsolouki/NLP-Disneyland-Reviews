import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def load_data(file_path):
    """
    Load the preprocessed data from a CSV file.
    """
    return pd.read_csv(file_path)


def vectorize_tfidf(data):
    """
    Vectorize the text data using TF-IDF.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_text'])
    return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()


def average_word_vectors(words, model, vocabulary, num_features):
    """
    Average the word vectors for a set of words.
    """
    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector


def vectorize_word2vec(data):
    """
    Vectorize the text data using a pre-trained Word2Vec model.
    """
    model = api.load("word2vec-google-news-300")
    vocabulary = set(model.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, 300) for tokenized_sentence in
                data['cleaned_text'].map(word_tokenize)]
    return pd.DataFrame(features)


from scipy.sparse import save_npz

if __name__ == "__main__":
    file_path = 'data/DisneylandReviews_cleaned.csv'
    data = load_data(file_path)

    # Vectorization using TF-IDF
    tfidf_matrix, tfidf_features = vectorize_tfidf(data)
    print("TF-IDF Vectorization completed.")
    save_npz('data/tfidf_matrix.npz', tfidf_matrix)

    # Vectorization using Word2Vec
    word2vec_matrix = vectorize_word2vec(data)
    print("Word2Vec Vectorization completed.")
    word2vec_matrix.to_csv('data/word2vec_matrix.csv', index=False)
