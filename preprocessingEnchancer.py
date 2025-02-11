import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


# Load spaCy model
nlp = spacy.load("data_set")
#TODO Implement text data

# Load text data
text = "This job requires knowledge of python and java"
tokens = word_tokenize(text)
tagged_tokens = nlp(text)

entities = [(ent.text, ent.label_)] for ent in tagged_tokens.ents]

def sentiment_analysis(text):
  # Creating an TextBlob object
  blob = TextBlob(text)
  sentiment = blob.sentiment
  return sentiment
  # TO use later
