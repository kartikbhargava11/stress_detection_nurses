# importing the libraries
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from .data import get_data, train_validation_split, features_and_labels_split
from .preprocessing import resample, normalization, sliding_window, series_to_multichannel
from .model import cnn_model


df = get_data()
training_data, validation_data = train_validation_split(df)
X, y = features_and_labels_split(training_data)

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# resampling has been employed. All the signals were resampled to 8 Hz.
X_train = resample(X_train)
X_test = resample(X_test)

# dataset has also been normalised to -1 to 1 interval
X_train = normalization(X_train)
X_test = normalization(X_test)

# overlapping time windows have been extracted from the input time series using sliding window approach and each window was converted to multichannel image representation i.e. 6 features for the given dataset. We used Gramian angular difference field for image representation. These images were classified using a convolutional neural network (CNN). These images were obtained from the given time series, these images represents temporal correlation between each time point

train_data = sliding_window(X_train, y_train)
test_data = sliding_window(X_test, y_test)

train_img = series_to_multichannel(train_data)
test_img = series_to_multichannel(test_data)


# model creation 
cnn_model(train_img, y_train, test_img, y_test)












