import preprocessed_dataset from  data_preprocessing
import tensorflow as tf


# Select % 75 of dataset
train_df = df_sample(frac=.075,, random_state = 4)

# Dropping training data from orig training set
val_df = df.drop(train_df.index)

def neural_network_model(train_df):
      





