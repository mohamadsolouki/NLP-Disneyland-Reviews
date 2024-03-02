import pandas as pd
from keybert import KeyBERT
from collections import Counter
from tqdm import tqdm
import pickle

# Load the cleaned reviews
df = pd.read_csv('data/cleaned_reviews.csv')
docs = df['Clean_Text'].tolist()

# Initialize KeyBERT with a specific model
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# Extract keywords from each document using KeyBERT with a progress bar
doc_keywords = []
for doc in tqdm(docs, desc='Extracting keywords', unit='doc'):
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 4), stop_words='english')
    doc_keywords.append(keywords)

# Save keybert keywords
df['Keywords'] = doc_keywords
df.to_csv('data/keybert_keywords.csv', index=False)

# save the model to disk using pickle
filename = 'models/keybert_model.sav'
pickle.dump(kw_model, open(filename, 'wb'))

# Aggregate keywords for the entire dataset
all_keywords = [word for sublist in doc_keywords for word, _ in sublist]
keyword_freq = Counter(all_keywords)

# Get the most common keywords across the dataset
common_keywords = keyword_freq.most_common(50)
print("Most common keywords across the dataset:")
for keyword, freq in common_keywords:
    print(f"{keyword}: {freq}")


if __name__ == "__main__":
    # This script can be run as a standalone program, with the above functions defined.
    pass
