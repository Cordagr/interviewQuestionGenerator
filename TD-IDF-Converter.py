import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset (Assuming CSV format)
df = pd.read_csv("interview_questions.csv")

# Display first few rows
print("Original Data:")
print(df.head())

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply cleaning to questions
df['cleaned_question'] = df['Question'].apply(clean_text)

# Convert to TF-IDF matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_question'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Save cleaned data
df.to_csv("cleaned_interview_questions.csv", index=False)
print("Cleaned Data Saved!")
