# Disneyland Reviews Complaint Analysis

This repository contains a series of scripts and analyses that focus on understanding customer complaints about Disneyland through Natural Language Processing (NLP) techniques. The objective is to preprocess the text data, vectorize it, and perform topic modeling to extract insights from the reviews.

## Project Structure

- `/data`: This directory contains the Disneyland reviews dataset used for the analysis.
- `/scripts`: Contains all the Python scripts used to process and analyze the data.
- `/images`: This directory stores visualizations created from the analyses, such as plots and word clouds.

## Scripts Overview

The analysis pipeline is broken down into four main Python scripts:

1. `eda.py`: An exploratory data analysis (EDA) script that generates initial observations, statistics, and visualizations to better understand the dataset.
2. `sentiment.py`: This script uses the TextBlob library to perform sentiment analysis on the reviews, providing insights into the overall sentiment of the customers.
2. `preprocess.py`: This script preprocesses the text data — cleaning, tokenizing, and filtering to prepare it for NLP tasks.
3. `bagofwords.py`: Utilizes the Bag of Words model to create a document-term matrix and analyze word frequencies within the dataset.
4. `tfidf.py`: Applies Term Frequency-Inverse Document Frequency (TF-IDF) to evaluate how important a word is to a document within a corpus.
5. `word2vec.py`: Uses the Word2Vec model to create word embeddings and analyze the semantic relationships between words.

## Dataset

The `data/` folder contains the Disneyland reviews dataset in CSV format. The dataset includes customer reviews which have been used to perform sentiment analysis, topic modeling, and other NLP tasks.

## Getting Started

To run this project, you will need to have Python installed on your system. You can then set up your environment using the following steps:

1. Clone this repository to your local machine using:
    ```
    git clone https://github.com/mohamadsolouki/NLP-Disneyland-Reviews.git
    ```
2. Navigate to the project directory: `cd NLP-Disneyland-Reviews`
3. Create and activate a virtual environment:
   - Windows: `python -m venv venv && .\venv\Scripts\activate`
   - macOS/Linux: `python -m venv venv && source venv/bin/activate`
4. Install the necessary Python packages listed in `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```
5. Navigate to the `scripts/` directory and execute the scripts in following order:
    ```
    python eda.py
    python sentiment.py
    python preprocess.py
    python bagofwords.py
    python tfidf.py
    python word2vec.py
    ```

## Visualization

The `images/` directory includes all visualizations generated during the analysis. These images provide insights into the dataset.

## Contributions
Contributions to this project are welcome. If you would like to contribute, please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file in the root directory for more details.

## Contact
For any further questions or comments, please feel free to open an issue in this repository, and I will get back to you as soon as possible.

I hope that this analysis provides valuable insights into Disneyland customer reviews and serves as a useful example of applying NLP techniques to real-world data.