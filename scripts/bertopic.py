import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load preprocessed data
df = pd.read_csv('data/cleaned_reviews.csv')

# Extract the reviews as a list
docs = df['Clean_Text'].tolist()

# Load a pre-trained sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create BERTopic model
topic_model = BERTopic(embedding_model=sentence_model, language="english", calculate_probabilities=True, verbose=True)

# Fit the BERTopic model
topics, probs = topic_model.fit_transform(docs)

# Get an overview of all the topics that the model has found
topic_info = topic_model.get_topic_info()

# Show the top 10 topics
for topic_num in topic_info['Topic'][:10]:
    if topic_num != -1:  # Exclude the -1 topic which contains outliers
        print(f"Topic {topic_num}: {topic_model.get_topic(topic_num)}\n")

# Save the model for future use
topic_model.save("models/bertopic_model")

# Save the topic modeling results to a CSV file for further analysis
df['Topic'] = topics
df['Topic_Probability'] = probs.max(axis=1)
df.to_csv('data/bertopic_results.csv', index=False)

# Visualize the topics
topic_model.visualize_topics()

# Visualize the topic barchart
topic_model.visualize_barchart(top_n_topics=16)

# Visualize the topic heatmap
topic_model.visualize_heatmap()
