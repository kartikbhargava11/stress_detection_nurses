import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField

import tensorflow as tf
from tensorflow import keras
from tslearn.preprocessing import TimeSeriesResampler
from keras.preprocessing.sequence import TimeseriesGenerator



def ts_to_img(X):
    # Convert the time series data into multichannel image representation
    gasf = GramianAngularField(image_size=X.shape[1]//2, method='difference')
    return gasf.fit_transform(X.reshape(-1, 1, X.shape[1]))

def resample(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Resample the data to 8 Hz
    # Resample the signals to 8 Hz
    resampler = TimeSeriesResampler(sz=8, interp_kind='linear')
    resampled_data = resampler.fit_transform(df.values)

def normalization(df):
    # Normalize the data to -1 to 1 interval
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df


def sliding_window(X, y):
    # Create time windows using sliding window approach
    window_size = pd.Timedelta(seconds=300) # 5 minutes
    batch_size = 32
    data = TimeseriesGenerator(X, y, length=window_size, batch_size=batch_size, shuffle=True)

    return data


def series_to_multichannel(series_data):
    # Convert the time series data to multichannel image representation
    img = np.array([ts_to_img(x)[0] for x, _ in series_data])

    # Reshape the data for the CNN input
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)

    return img

    



