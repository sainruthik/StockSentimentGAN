import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')

def clean_tweet(tweet):
    """
    Cleans tweet text by removing URLs, mentions, hashtags, and special characters.

    Parameters:
    - tweet: The raw tweet text.

    Returns:
    - Cleaned tweet text.
    """
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)     # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)     # Remove hashtags
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)  # Remove special characters
    tweet = tweet.lower()  # Convert to lowercase
    return tweet

def analyze_sentiment(tweets_df, date_column='Date', stock_column='Stock Name', tweet_column='Tweet'):
    """
    Analyzes sentiment of tweets and aggregates by Date and Stock Name.

    Parameters:
    - tweets_df: DataFrame containing tweets.
    - date_column: Column name for dates.
    - stock_column: Column name for stock names.
    - tweet_column: Column name for tweet texts.

    Returns:
    - DataFrame with aggregated mean sentiment scores per Date and Stock Name.
    """
    sia = SentimentIntensityAnalyzer()
    
    # Clean tweets
    tweets_df['Cleaned_Tweet'] = tweets_df[tweet_column].apply(clean_tweet)
    
    # Calculate sentiment scores
    tweets_df['Sentiment_Score'] = tweets_df['Cleaned_Tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Remove the time component from the 'Date' column
    tweets_df[date_column] = tweets_df[date_column].dt.date
    
    # Group by Date and Stock Name, and calculate the mean sentiment score
    sentiment_agg = tweets_df.groupby([date_column, stock_column]).agg({'Sentiment_Score': 'mean'}).reset_index()
    
    # Rename the column to 'Sentiment' for clarity
    sentiment_agg.rename(columns={'Sentiment_Score': 'Sentiment'}, inplace=True)
    
    return sentiment_agg

if __name__ == "__main__":
    # Path to the CSV file containing tweets
    tweets_path = './Dataset/stock_tweets.csv'  # Replace with your actual file path
    output_path = './Dataset/processed_sentiment_scores.csv'
    
    # Load tweets from CSV
    tweets_df = pd.read_csv(tweets_path, parse_dates=['Date'])
    
    # Perform sentiment analysis
    sentiment_scores = analyze_sentiment(tweets_df)
    
    # Save the sentiment scores to a CSV file, aggregated by Date and Stock Name
    sentiment_scores.to_csv(output_path, index=False)
    print(f"Sentiment scores saved to {output_path}")
