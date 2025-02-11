import kagglehub
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Add more stop words
additional_stop_words = {'also', 'would', 'could', 'should', 'may', 'might', 'must', 'need', 'want', 'try', 'one', 'two', 'three', 'using', 'use', 'used', 'way', 'ways'}
stop_words = stop_words.union(additional_stop_words)

# Download latest dataset
path = kagglehub.dataset_download("syedmharis/software-engineering-interview-questions-dataset")
print("Path to dataset files:", path)

# Load dataset (Assuming CSV format)
df = pd.read_csv(f"{path}/interview_questions.csv")

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
