import re
import spacy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')

# read in the file path
dir_path = '../../CS6220-NFT-Celebrities-Twitter-Analysis-main/tweets_by_date'
res = []
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)

# clean tweets
def clean_tweet(tweet):
    # Convert to string if not a string
    tweet = str(tweet)
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove user mentions
    tweet = re.sub(r"@\w+", "", tweet)
    # Remove special characters
    tweet = re.sub(r"\W", " ", tweet)
    # Remove extra spaces
    tweet = re.sub(r"\s+", " ", tweet)
    return tweet.strip()

# Load the stopwords and stemmer once
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
# tokenize
def preprocess_tweet(tweet):
    # Tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False)
    tokens = tokenizer.tokenize(tweet)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatized_tokens = []
    for token in tokens:
        doc = nlp(token)
        lemmatized_tokens.append([t.lemma_ for t in doc][0])

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    return stemmed_tokens

# VADER
def analyze_sentiment_vader(tweet):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(' '.join(tweet))
    return sentiment_score

# process by date
for i in tqdm(res):
    path = '../../CS6220-NFT-Celebrities-Twitter-Analysis-main/tweets_by_date/{}'.format(i)
    tweets = pd.read_csv(path)
    tweets['clean_text'] = tweets['Text'].apply(clean_tweet)
    tweets['tokenize_text'] = tweets['clean_text'].apply(preprocess_tweet)

    # VADER
    tweets['vader'] = tweets['tokenize_text'].apply(analyze_sentiment_vader)
    tweets.to_csv('../../CS6220-NFT-Celebrities-Twitter-Analysis-main/tweet_after_clean_tokenize/vader/vader_{}'.format(i))
