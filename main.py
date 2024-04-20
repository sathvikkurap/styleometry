import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, smog_index, coleman_liau_index
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import string

nltk.download('stopwords')

# Load the text of "Kindling" into a string
with open('kindling.txt', 'r') as file:
    text = file.read()

# Tokenize the text into words
words = word_tokenize(text)

# Clean the text by removing non-alphabetic characters and converting to lowercase
clean_words = [word.lower() for word in words if word.isalpha()]

# Remove stopwords
stop_words = set(stopwords.words('english'))
clean_words = [word for word in clean_words if word not in stop_words]

# Calculate word frequency
word_freq = FreqDist(clean_words)

# Calculate average word length
avg_word_length = sum(len(word) for word in clean_words) / len(clean_words)

# Tokenize the text into sentences using spaCy for better accuracy
nlp = spacy.load('en_core_web_sm')
sentences = [sent.text for sent in nlp(text).sents]

# Calculate average sentence length
avg_sentence_length = len(clean_words) / len(sentences)

# Calculate vocabulary richness (unique words / total words)
vocabulary_richness = len(set(clean_words)) / len(clean_words)

# Calculate punctuation frequency
punctuation_freq = FreqDist([char for char in text if char in string.punctuation])

# POS tagging using spaCy
pos_tags = [(token.text, token.pos_) for token in nlp(text)]
noun_count = len([word for word, pos in pos_tags if pos == 'NOUN'])

# Sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [sentiment_analyzer.polarity_scores(sent)['compound'] for sent in sentences]
avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

# Named Entity Recognition (NER) using spaCy
named_entities = Counter([ent.text for ent in nlp(text).ents])

# Calculate readability scores
flesch_score = flesch_reading_ease(text)
smog_score = smog_index(text)
coleman_score = coleman_liau_index(text)

# Plot word frequency distribution
plt.figure(figsize=(12, 6))
plt.bar([word for word, _ in word_freq.most_common(20)], [freq for _, freq in word_freq.most_common(20)])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Word Frequency in "Kindling"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot sentence length distribution
plt.figure(figsize=(8, 6))
plt.hist([len(sent.split()) for sent in sentences], bins=20)
plt.xlabel('Sentence Length (in words)')
plt.ylabel('Frequency')
plt.title('Sentence Length Distribution in "Kindling"')
plt.show()

# Plot POS tag distribution
pos_counts = Counter([pos for _, pos in pos_tags])
plt.figure(figsize=(8, 6))
plt.bar(pos_counts.keys(), pos_counts.values())
plt.xlabel('POS Tags')
plt.ylabel('Frequency')
plt.title('POS Tag Distribution in "Kindling"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the results
print(f"Total Words: {len(clean_words)}")
print(f"Unique Words: {len(set(clean_words))}")
print(f"Average Word Length: {avg_word_length:.2f}")
print(f"Average Sentence Length: {avg_sentence_length:.2f}")
print(f"Vocabulary Richness: {vocabulary_richness:.2f}")
print(f"Punctuation Frequency: {punctuation_freq}")
print(f"Noun Count: {noun_count}")
print(f"Average Sentiment Score: {avg_sentiment_score:.2f}")
print(f"Named Entities: {named_entities}")
print(f"Flesch Reading Ease Score: {flesch_score:.2f}")
print(f"SMOG Index: {smog_score:.2f}")
print(f"Coleman-Liau Index: {coleman_score:.2f}")
