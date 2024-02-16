import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from scipy.sparse import load_npz
import pickle


def load_vectorized_data(tfidf_path, word2vec_path):
    """
    Load the vectorized data from files.
    """
    tfidf_matrix = load_npz(tfidf_path)
    word2vec_matrix = pd.read_csv(word2vec_path)
    return tfidf_matrix, word2vec_matrix


def lda_model(tfidf_matrix, num_topics=5):
    """
    Apply LDA topic modeling to the TF-IDF matrix.
    """
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)
    return lda


def lsa_model(word2vec_matrix, num_topics=5):
    """
    Apply LSA topic modeling to the Word2Vec matrix.
    """
    lsa = TruncatedSVD(n_components=num_topics)
    lsa.fit(word2vec_matrix)
    return lsa


def save_model(model, filename):
    """
    Save the model to a file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    tfidf_path = 'data/tfidf_matrix.npz'
    word2vec_path = 'data/word2vec_matrix.csv'

    tfidf_matrix, word2vec_matrix = load_vectorized_data(tfidf_path, word2vec_path)

    # Apply LDA to TF-IDF
    lda = lda_model(tfidf_matrix)
    print("LDA Model: ", lda.components_)

    # Apply LSA to Word2Vec
    lsa = lsa_model(word2vec_matrix)
    print("LSA Model: ", lsa.components_)

    # Save the models for later analysis
    save_model(lda, 'models/lda_model.pkl')
    save_model(lsa, 'models/lsa_model.pkl')
