import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# loading spaCy model
nlp = spacy.load('en_core_web_sm')

# Adding spacytextblob extension
nlp.add_pipe('spacytextblob')

# Preprocessing  with tokenization and removing stopwords and basic cleaning
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct])

# using Sentiment analysis function
def analyze_sentiment(text):
    doc = nlp(text)
    polarity = doc._.blob.polarity      # polarity score
    sentiment = doc._.blob.sentiment    # sentiment score
    label = "positive" if 0.1 <polarity <1 else ("negative" if -1 <polarity < -0.1 else "neutral")  # Labeling sentiment
    return {"polarity": polarity, "sentiment": sentiment, "label": label} 


# removing missing datas 
df = pd.read_csv('amazon_product_reviews.csv', sep=',')
df = df.dropna(subset=['reviews.text'])  

# taking only the reviews column in
df = df[['reviews.text']]

# Selecting random 10 reviews
sample_reviews = df.sample(10, random_state=10)

# Applying preprocessing to the reviews
sample_reviews['processed_reviews'] = sample_reviews['reviews.text'].apply(preprocess)

# Analyzing sentiments
sentiment_analysis = []
for review in sample_reviews['processed_reviews']:
    sentiment_analysis.append(analyze_sentiment(review))

sample_reviews['polarity'] = [analysis['polarity'] for analysis in sentiment_analysis]
sample_reviews['sentiment'] = [analysis['sentiment'] for analysis in sentiment_analysis]
sample_reviews['label'] = [analysis['label'] for analysis in sentiment_analysis]

print("\nSentiment Analysis Results")
print(sample_reviews[['reviews.text', 'polarity', 'sentiment', 'label']])


# Loading medium spacy model
nlp = spacy.load("en_core_web_md")

# Two reviews to compare
review_of_choice_1 = df['reviews.text'][40]
review_of_choice_2 = df['reviews.text'][4050]

# Preprocessing the reviews using spaCy
review1_doc = nlp(review_of_choice_1)
review2_doc = nlp(review_of_choice_2)

# Computing the similarity score between the two reviews
similarity_score = review1_doc.similarity(review2_doc)

# Printing the similarity score
print("\nSimilarity Score")
print("Similarity Score:", similarity_score)
