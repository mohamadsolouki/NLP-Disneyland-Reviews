import pandas as pd
from keybert import KeyBERT
from collections import Counter
from tqdm import tqdm

# Load the cleaned reviews
df = pd.read_csv('data/cleaned_reviews.csv')
docs = df['Clean_Text'].tolist()

# Initialize KeyBERT with a specific model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# Extract keywords from each document using KeyBERT with a progress bar
doc_keywords = []
for doc in tqdm(docs, desc='Extracting keywords', unit='doc'):
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english')
    doc_keywords.append(keywords)

# Aggregate keywords for the entire dataset
all_keywords = [word for sublist in doc_keywords for word, _ in sublist]
keyword_freq = Counter(all_keywords)

# Get the most common keywords across the dataset
common_keywords = keyword_freq.most_common(30)
print("Most common keywords across the dataset:")
for keyword, freq in common_keywords:
    print(f"{keyword}: {freq}")

# Concatenate documents to create a single text corpus
corpus = ' '.join(docs)

# Extract keywords from the concatenated corpus
corpus_keywords = kw_model.extract_keywords(corpus, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)

# Print the extracted keywords from the concatenated corpus
print("\nKeywords representative of the entire corpus:")
for keyword, score in corpus_keywords:
    print(f"{keyword}: {score}")