import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy


# Load the dataset
file_path = 'data/DisneylandReviews.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load English tagger, and word vectors
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Custom stopwords
custom_stopwords = {'disney', 'land', 'disneyland', 'go', 'one', 'kid', 'world', 'I', 'day', 'thing', 'went', 'child',
                    'daughter'}
stop_words = set(stopwords.words('english')) | custom_stopwords

# Lemmatizer
lemmatizer = WordNetLemmatizer()


# Handling negations by creating bi-grams with negation word and subsequent word.
def handle_negations(text):
    # Define the negation pattern
    negation_pattern = re.compile(
        r"\b(not|no|never|none|cannot|can't|couldn't|shouldn't|won't|wouldn't|don't|doesn't|didn't|isn't|aren't|ain't"
        r")\s([a-z]+)\b",
        re.IGNORECASE
    )
    negated_form = r'\1_\2'  # E.g., "not_good"
    return negation_pattern.sub(negated_form, text)


# Function to preprocess text
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = handle_negations(text)  # Handle negations
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        lemmatized = nlp(' '.join(tokens))
        lemmatized = [token.lemma_ for token in lemmatized]
        return ' '.join(lemmatized)
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""


# Apply preprocessing to the Review_Text column of the DataFrame
low_rating_threshold = 3
df = df[df['Rating'] <= low_rating_threshold]
df['Clean_Text'] = df['Review_Text'].apply(preprocess_text)

# Export to a new CSV file
df.to_csv('data/cleaned_reviews.csv', index=False)

if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass
