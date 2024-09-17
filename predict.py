import numpy as np
import pickle
from textblob import TextBlob

# Load the generator model
def load_generator():
    with open('./models/epoch_final/generator.pkl', 'rb') as f:
        generator = pickle.load(f)
    return generator

generator = load_generator()

# Perform sentiment analysis on the input tweet
def perform_sentiment_analysis(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity  # Sentiment score (range: -1 to 1)

# Generate stock price predictions using the GAN model
def generate_predictions(generator, noise_dim, sentiment_score):
    noise = np.random.normal(0, 1, (1, noise_dim - 1))  # Adjust noise_dim to fit sentiment
    noise_with_sentiment = np.concatenate((noise, np.array([[sentiment_score]])), axis=1)
    generated_samples = generator.predict(noise_with_sentiment)
    return generated_samples[0][0]

# Main function for predicting stock price based on company and tweet
def predict_stock(company, tweet):
    sentiment_score = perform_sentiment_analysis(tweet)
    noise_dim = 10  # Adjust based on your generator's input size
    predicted_price = generate_predictions(generator, noise_dim, sentiment_score)
    return predicted_price, sentiment_score
