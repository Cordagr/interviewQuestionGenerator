import torch
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

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

os.environ["TOKENIZERS_PARALLELISM"] = "false"

task = "mrpc"
task_meta = glue_tasks[task]
train_ds_name = task_meta["dataset_names"]["train"]
valid_ds_name = task_meta["dataset_names"]["valid"]
test_ds_name = task_meta["dataset_names"]["test"]

task_inputs = task_meta["inputs"]
task_target = task_meta["target"]
n_labels = task_meta["n_labels"]
task_metrics = task_meta["metric_funcs"]

checkpoint = "answerdotai/ModernBERT-base"  # "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"

#Load dataset
dataset = load.datasets("processed_data",task)
# Loading predefined model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",  problem_type="multi_label_classification", num_labels=len(labels),id2label=id2label,label2id=label2id)



def get_label_question_label_map(datasets_, train_ds_name):
    # Extract features

    key_word_to_questions = defaultdict(list)
    # Define a broad list of programming languages & skills
    skill_keywords = {
        "Python": r"\b(python|pandas|numpy|flask|django)\b",
        "Java": r"\b(java|spring|hibernate|jvm|junit)\b",
        "C++": r"\b(c\+\+|stl|boost|cpp)\b",
        "JavaScript": r"\b(javascript|js|node\.js|react|angular|vue)\b",
        "SQL": r"\b(sql|database|postgresql|mysql|joins|queries)\b",
        "Machine Learning": r"\b(machine learning|ml|neural network|tensorflow|pytorch|deep learning)\b",
        "Data Structures & Algorithms": r"\b(binary search|linked list|graph|tree|sorting|recursion)\b"
    }

    topics_to_assigned_questions = defaultdict(list)

    for question in dataset:
        # Regex up to Difficulty and categorize differently
        question_text = question
    










