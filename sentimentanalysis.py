import pandas as pd
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the Amazon product reviews dataset
data = pd.read_csv("amazon_product_review.csv")


# Cleaning data: Remove stopwords and clean text
def preprocess_text(text):
    doc = nlp(text)
    # Remove stopwords and punctuation
    reviews_data = data['reviews.text']
    clean_tokens = data.dropna(subset=['reviews.text'])
    clean_tokens = [token.text.lower().strip() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(clean_tokens)

# Applying text pre-processing to the review textt column
data['clean_review_text'] = data['reviews.text'].apply(preprocess_text)
# Load the spaCy model
from textblob import TextBlob
nlp = spacy.load("en_core_web_md")
def analyze_polarity(text):
    # Preprocess the text with spaCy
    doc = nlp(text)
    
    # Analyze sentiment with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity
    
# Testing the sentiment analysis function on a few sample product reviews
test1 = "This product is amazing! I love it."
test2 = "I'm not satisfied with this product. It's terrible."
test3 =  "Average product. Nothing special."
polarity_score1 = analyze_polarity(test1)
polarity_score2 = analyze_polarity(test2)
polarity_score3 = analyze_polarity(test3)

if polarity_score > 0:
    sentiment = 'positive'
elif polarity_score < 0:
    sentiment = 'negative'
else:
    sentiment = 'neutral'
print(f"Test: {test1}\nPolarity score: {polarity_score1}\nSentiment: {sentiment}")
print(f"Test: {test2}\nPolarity score: {polarity_score2}\nSentiment: {sentiment}")
print(f"Test: {test3}\nPolarity score: {polarity_score3}\nSentiment: {sentiment}")

# -1 to 1
