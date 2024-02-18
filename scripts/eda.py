# EDA.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_missing_values(data):
    """
    Check for missing values in the dataset.

    """
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_percentage = (missing_values / len(data)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_values, 'Missing Percentage': missing_percentage})
    return missing_info


def perform_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the Disneyland reviews dataset.
    
    Args:
        data (pandas.DataFrame): The dataset to analyze.
    
    Returns:
        dict: A dictionary containing EDA results and visualizations.
    """
    eda_results = {
        'missing_values': check_missing_values(data),
        'missing_year_month': (data['Year_Month'] == 'missing').sum(),
        'rating_distribution': data['Rating'].value_counts()
    }

    # Visualization 1: Rating Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x='Rating', hue='Rating', dodge=False, legend=False)
    plt.title('Rating Distribution')
    plt.ylabel('Count')
    plt.xlabel('Rating')
    plt.tight_layout()
    plt.savefig('images/rating_distribution.png')
    plt.show()
    eda_results['rating_distribution_plot'] = plt

    # Visualization 2: Reviews per Disneyland Branch
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, y='Branch', hue='Branch', dodge=False, order=data['Branch'].value_counts().index,
                  legend=False)
    plt.title('Reviews per Disneyland Branch')
    plt.xlabel('Count')
    plt.ylabel('Branch')
    plt.tight_layout()
    plt.savefig('images/branch_distribution.png')
    plt.show()
    eda_results['branch_distribution_plot'] = plt

    # Visualization 3: Review Length Distribution
    data['Review_Length'] = data['Review_Text'].apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(data['Review_Length'], bins=50, kde=True, color='blue')
    plt.title('Review Length Distribution')
    plt.xlabel('Length of Review')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('images/review_length.png')
    plt.show()
    eda_results['review_length_plot'] = plt

    # Visualization 4: Reviews per Year
    data['Year'] = data['Year_Month'].apply(lambda x: x.split('-')[0] if x != 'missing' else 'Unknown')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Year', hue='Year', dodge=False, order=sorted(data['Year'].unique()), legend=False)
    plt.title('Reviews per Year')
    plt.ylabel('Count')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/year_distribution.png')
    plt.show()
    eda_results['year_distribution_plot'] = plt

    # Visualization 5: Reviews per Reviewer Location (Top 10)
    top_locations = data['Reviewer_Location'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_locations, y=top_locations.index, hue=top_locations.index, dodge=False, legend=False)
    plt.title('Top 10 Reviewer Locations')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Reviewer Location')
    plt.tight_layout()
    plt.savefig('images/top_locations.png')
    plt.show()
    eda_results['top_locations_plot'] = plt

    # Visualization 6: Rating Over Years by Branch
    rating_per_year_branch = df.groupby(['Year', 'Branch'])['Rating'].mean().unstack()
    plt.figure(figsize=(14, 8))
    rating_per_year_branch.plot(marker='o', linewidth=2)
    plt.title('Average Rating Over Years by Branch')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.legend(title='Branch', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/rating_years_branch.png')
    plt.show()
    eda_results['rating_years_branch'] = plt

    # Visualization 7: Wordcloud for positive reviews with 4 or 5 rating
    from wordcloud import WordCloud
    positive_reviews = data[data['Rating'] >= 4]
    positive_reviews_text = ' '.join(positive_reviews['Review_Text'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white',
                          max_words=100).generate(positive_reviews_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Wordcloud for Positive Reviews')
    plt.tight_layout()
    plt.savefig('images/wordcloud_positive_reviews.png')
    plt.show()
    eda_results['wordcloud_positive_reviews'] = plt

    # Visualization 8: Wordcloud for negative reviews with 1, 2 or 3 rating
    negative_reviews = data[data['Rating'] <= 3]
    negative_reviews_text = ' '.join(negative_reviews['Review_Text'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white',
                          max_words=100).generate(negative_reviews_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Wordcloud for Negative Reviews')
    plt.tight_layout()
    plt.savefig('images/wordcloud_negative_reviews.png')
    plt.show()
    eda_results['wordcloud_negative_reviews'] = plt

    return eda_results


if __name__ == '__main__':
    # Update the file path if necessary
    file_path = 'data/DisneylandReviews.csv'
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    eda_results = perform_eda(df)
    for key, value in eda_results.items():
        if isinstance(value, plt.Figure):
            value.show()
