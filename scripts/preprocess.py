import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess a single text string.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

def preprocess_data(file_path):
    """
    Load, preprocess, and save the dataset.
    """
    # Load data
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Preprocess text data
    data['cleaned_text'] = data['Review_Text'].apply(preprocess_text)

    # Save the cleaned data
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")

if __name__ == "__main__":
    file_path = 'data/DisneylandReviews.csv'
    preprocess_data(file_path)
