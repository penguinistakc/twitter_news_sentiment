# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_tweets(tweet):
    # Run Vader Analysis on each tweet
    return analyzer.polarity_scores(tweet["text"])