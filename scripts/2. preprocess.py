import nltk
import spacy
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load English tokenizer, tagger, parser, NER, and word vectors
# Disabling unnecessary components for efficiency
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Custom stopwords
custom_stopwords = {
    'disney', 'land', 'disneyland', 'rides', 'ride', 'good', 'really', 'very', 'quite',
    'pretty', 'especially', 'actually', 'probably', 'maybe', 'sure', 'time', 'day', 'year',
    'thing', 'world', 'point', 'bit', 'number', 'week', 'make', 'say', 'come', 'go', 'know',
    'take', 'see', 'get', 'want', 'think', 'look', 'tell', 'try', 'use', 'need', 'feel',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
    'your', 'his', 'its', 'our', 'their', 'a', 'an', 'the', 'in', 'on', 'at', 'from', 'with',
    'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'over', 'under', 'to', 'of', 'for', 'by', 'and', 'but', 'or', 'so', 'yet', 'because',
    'as', 'until', 'than', '10', '20', '30', '45', '15', 'minute', 'second', 'hour', 'day', 'pm'
    'park', 'go', 'one', 'kid'
}

# Update stop words list
stop_words = set(stopwords.words('english')).union(custom_stopwords)

# Filtering the DataFrame to include only rows with low ratings
df = pd.read_csv('data/DisneylandReviews.csv', encoding='ISO-8859-1')
low_rating_threshold = 3
clean_df = df[df['Rating'] <= low_rating_threshold]

# Handling negations by creating bi-grams with negation word and subsequent word.
def handle_negations(text):
    # Define the negation pattern
    negation_pattern = re.compile(
        r"\b(not|no|never|none|cannot|can't|couldn't|shouldn't|won't|wouldn't|don't|doesn't|didn't|isn't|aren't|ain't)\s([a-z]+)\b",
        re.IGNORECASE
    )
    negated_form = r'\1_\2'  # E.g., "not_good"
    return negation_pattern.sub(negated_form, text)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Normalize text to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = handle_negations(text)  # Handle negations
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    lemmatized = nlp(' '.join(tokens))  # Lemmatization
    lemmatized = [token.lemma_ for token in lemmatized]
    return ' '.join(lemmatized)

# Apply preprocessing to the Review_Text column of the DataFrame
clean_df['Clean_Text'] = clean_df['Review_Text'].apply(preprocess_text)

# Display the first few rows of the processed data
print(clean_df[['Review_Text', 'Clean_Text']].head())

# Export to a new CSV file
clean_df.to_csv('data/cleaned_reviews.csv', index=False)

# Function to display a word cloud
def show_wordcloud(text):
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/wordcloud.png')
    plt.show()

# Create a word cloud for all reviews
all_reviews = ' '.join(clean_df['Clean_Text'])
show_wordcloud(all_reviews)