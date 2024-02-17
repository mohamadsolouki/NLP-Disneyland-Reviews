from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load the preprocessed reviews
df = pd.read_csv('data/DisneylandReviews.csv', encoding='ISO-8859-1')

# calculate sentiment and put it in a new column
df['sentiment'] = df['Review_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# convert sentiment to positive or negative and put it in a new column
df['sentiment_cat'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Save the sentiment analysis results
df.to_csv('data/sentiment_analysis.csv', index=False)

# Print the average sentiment of the reviews
print("The average sentiment of the reviews is:")
print(df['sentiment'].mean())

# Print the standard deviation of the sentiment of the reviews
print("The standard deviation of the sentiment of the reviews is:")
print(df['sentiment'].std())

# plot the sentiment distribution
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], kde=True, color='blue')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
plt.savefig('images/sentiment_distribution.png')
