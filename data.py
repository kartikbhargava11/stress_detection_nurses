# importing the libraries
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# reading dataset
def get_data():
    return pd.read_csv("merged_data_labeled.csv", dtype={'id': str, 'label': int})


# splits data into training and validation set
def train_validation_split(df):
    ids = list(df["id"].value_counts().index)
    # randomly selected 12 participants for model training
    selected_ids = random.sample(ids, 12)
    training_data = df[df['id'] in selected_ids]

    # remaining 3 participants for validation data
    validation_data = df[df['id'] not in selected_ids]

    return training_data, validation_data

def features_and_labels_split(df):
    X = df[df.columns[:-1]].values
    y = df['label'].values
    return X, y

