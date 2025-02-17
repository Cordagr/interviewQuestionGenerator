import torch
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from transformers import pipeline, AutoModelForSequenceClassification
from pprint import pprint
import numpy as np 
import pandas as pd
import gc
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


def parse_and_categorize_questions(dataset_path):
    """
    Parses the dataset and categorizes questions by topic and difficulty.
    
    :param dataset_path: Path to the dataset (CSV or text file).
    :return: Dictionary {Topic: {Difficulty: [Questions]}}
    """
    # Load dataset
    df = pd.read_csv(dataset_path, header=None, names=["ID", "Question", "Answer", "Topic", "Difficulty"])

    # Define regex patterns for keyword matching
    skill_keywords = {
        "Python": r"\b(python|pandas|numpy|flask|django)\b",
        "Java": r"\b(java|spring|hibernate|jvm|junit)\b",
        "C++": r"\b(c\+\+|stl|boost|cpp)\b",
        "JavaScript": r"\b(javascript|js|node\.js|react|angular|vue)\b",
        "SQL": r"\b(sql|database|postgresql|mysql|joins|queries)\b",
        "Machine Learning": r"\b(machine learning|ml|neural network|tensorflow|pytorch|deep learning)\b",
        "OOP": r"\b(polymorphism|inheritance|encapsulation|abstraction|interface|class)\b",
        "Data Structures": r"\b(binary search|linked list|graph|tree|sorting|recursion)\b"
    }

    # Dictionary to store categorized questions
    categorized_questions = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        question = row["Question"]
        topic = row["Topic"]
        difficulty = row["Difficulty"]

        # Detect additional skills from question text
        matched_skills = [skill for skill, pattern in skill_keywords.items() if re.search(pattern, question.lower())]
        
        # If no specific skill is matched, keep the dataset topic
        if not matched_skills:
            matched_skills.append(topic)

        # Store questions under the correct topic and difficulty
        for skill in matched_skills:
            categorized_questions[skill][difficulty].append(question)

    return categorized_questions

# Example Usage:
dataset_path = "questions_dataset.csv"  # Update with the correct file path
categorized_data = parse_and_categorize_questions(dataset_path)

# Print results (example output)
pprint.pprint(categorized_data)

# Below is strucutured format after parsing and cartegorizing 
"""{
    'General Programming': {
        'Medium': ['What is the difference between compilation and interpretation?']
    },
    'OOP': {
        'Medium': ['Explain polymorphism.', 'What is an abstract class?']
    },
    'SQL': {
        'Medium': ['Describe SQL joins.']
    },
    'Data Structures': {
        'Medium': ['What is a linked list?']
    }
} """

# Loading predefined model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",  problem_type="multi_label_classification", num_labels=len(labels),id2label=id2label,label2id=label2id)





