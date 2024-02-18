from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# load the preprocessed reviews
df = pd.read_csv('data/DisneylandReviews.csv', encoding='ISO-8859-1')

# calculate sentiment and put it in a new column
df['sentiment'] = df['Review_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# convert sentiment to positive or negative and put it in a new column
df['sentiment_cat'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Save the sentiment analysis results
df.to_csv('data/sentiment_analysis.csv', index=False)

# plot the sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], kde=True, color='skyblue')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/sentiment_distribution.png')
plt.show()

# plot the sentiment distribution by category
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment_cat', hue='sentiment_cat', dodge=False, palette='pastel')
plt.title('Sentiment Distribution by Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/sentiment_category_distribution.png')
plt.show()

# Sentiment Analysis using Vader
# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()

# Define a function to get the sentiment score
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

# Calculate the sentiment score and put it in a new column
df['sentiment_vader'] = df['Review_Text'].apply(get_sentiment_score)

# Convert sentiment to positive or negative and put it in a new column
df['sentiment_cat_vader'] = df['sentiment_vader'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Compare the sentiment scores from TextBlob and vaderSentiment
print("The average sentiment score from TextBlob is:")
print(df['sentiment'].mean())
print("The average sentiment score from vaderSentiment is:")
print(df['sentiment_vader'].mean())

# Compare the sentiment categories from TextBlob and vaderSentiment
print("The sentiment category distribution from TextBlob is:")
print(df['sentiment_cat'].value_counts())
print("The sentiment category distribution from vaderSentiment is:")
print(df['sentiment_cat_vader'].value_counts())

# Compare the standard deviation of the sentiment scores from TextBlob and vaderSentiment
print("The standard deviation of the sentiment scores from TextBlob is:")
print(df['sentiment'].std())
print("The standard deviation of the sentiment scores from vaderSentiment is:")
print(df['sentiment_vader'].std())

# plot the sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_vader'], kde=True, color='skyblue')
plt.title('Sentiment Distribution (vaderSentiment)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/sentiment_distribution_vader.png')
plt.show()

# plot the sentiment distribution by category
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment_cat_vader', hue='sentiment_cat_vader', dodge=False, palette='pastel')
plt.title('Sentiment Distribution by Category (vaderSentiment)')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/sentiment_category_distribution_vader.png')
plt.show()


# Save the sentiment analysis results
df.to_csv('data/sentiment_analysis.csv', index=False)