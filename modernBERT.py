import torch
from transformers import pipeline
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















