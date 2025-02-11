import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model (replace "data_set" with the correct model name)
try:
    nlp = spacy.load("en_core_web_sm")  # Or a larger model like "en_core_web_lg"
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load stop words
stop_words = set(stopwords.words('english'))

# Add more stop words
additional_stop_words = {'also', 'would', 'could', 'should', 'may', 'might', 'must', 'need', 'want', 'try', 'one', 'two', 'three', 'using', 'use', 'used', 'way', 'ways'}
stop_words = stop_words.union(additional_stop_words)

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
    tokens = nltk.word_tokenize(text)  # Use nltk.word_tokenize
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

# Download latest dataset
path = kagglehub.dataset_download("syedmharis/software-engineering-interview-questions-dataset")
print("Path to dataset files:", path)

# Load dataset (Assuming CSV format)
df = pd.read_csv(f"{path}/interview_questions.csv")

# Display first few rows
print("Original Data:")
print(df.head())

# Apply cleaning to questions
df['cleaned_question'] = df['Question'].apply(clean_text)

# Grouping (Example: Group by question category if available)
# Assuming you have a 'Category' column in your DataFrame
if 'Category' in df.columns:
    grouped = df.groupby('Category')['cleaned_question'].apply(lambda x: ' '.join(x))
    print("Grouped Data (by Category):")
    print(grouped)
else:
    print("No 'Category' column found. Skipping grouping.")
    grouped = None  # Or handle the case where grouping isn't possible

# Convert to TF-IDF matrix
if grouped is not None:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(grouped)

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
else:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_question'])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Save cleaned data
df.to_csv("cleaned_interview_questions.csv", index=False)
print("Cleaned Data Saved!")
