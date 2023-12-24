import nltk
import pandas as pd
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load Spacy's English-language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# NLTK Stop words
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    """
    Preprocess a single text string.
    """
    # Convert to lowercase
    text = text.lower()

    # Handle negations by appending '_NEG' to words following a negation
    negation_pattern = re.compile(r'\b(?:not|no|never|none|nobody|nothing|neither|nowhere|hardly|scarcely|barely'
                                  r'|doesn’t|isn’t|wasn’t|shouldn’t|wouldn’t|couldn’t|won’t|can’t|don’t)\b[\w\s]+['
                                  r'^\w\s]')
    negated_phrases = negation_pattern.findall(text)
    for phrase in negated_phrases:
        negated_text = re.sub(r'\s+', '_NEG ', phrase)
        text = text.replace(phrase, negated_text)

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatized = [token.lemma_ for token in nlp(" ".join(tokens))]

    return ' '.join(lemmatized)


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
