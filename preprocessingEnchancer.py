import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re
import string

# Load spaCy model (replace "data_set" with the correct model name)
try:
    nlp = spacy.load("en_core_web_sm")  # Or a larger model like "en_core_web_lg"
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load stop words
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing punctuation.
    3. Removing stop words.
    4. Lemmatizing words.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[{}]".format(string.punctuation), "", text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def extract_entities(text):
    """
    Extracts named entities from the text using spaCy.

    Args:
        text (str): The text to analyze.

    Returns:
        list: A list of tuples, where each tuple contains the entity text and its label.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def sentiment_analysis(text):
    """
    Performs sentiment analysis on the given text using TextBlob.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the polarity, subjectivity, and overall sentiment label.
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = sentiment.polarity  # Range: -1 (negative) to 1 (positive)
    subjectivity = sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)

    if polarity > 0.1:
        sentiment_label = "Positive"
    elif polarity < -0.1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment_label
    }


# Load text data
text = "This job requires knowledge of python and java.  It's a challenging but rewarding opportunity!"

# Clean the text
cleaned_text = clean_text(text)
print(f"Cleaned Text: {cleaned_text}")

# Extract entities
entities = extract_entities(text)  # Use original text for entity extraction
print(f"Entities: {entities}")

# Perform sentiment analysis
sentiment = sentiment_analysis(cleaned_text)  # Use cleaned text for sentiment
print(f"Sentiment: {sentiment}")
