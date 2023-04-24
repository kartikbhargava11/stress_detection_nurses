import os
import shutil
import multiprocessing
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from functools import partial

def read_parallel(COMBINED_DATA_PATH, signal):
    print(signal)
    df = pd.read_csv(os.path.join(COMBINED_DATA_PATH, f"combined_{signal}.csv"), dtype={'id': str})
    return [signal, df]


def merge_parallel(acc, eda, hr, temp, columns, id):
    print(f"Processing {id}")
    df = pd.DataFrame(columns=columns)
    acc_id = acc[acc['id'] == id]
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)
    df = acc_id.merge(eda_id, on='datetime', how='outer')
    df = df.merge(temp_id, on='datetime', how='outer')
    df = df.merge(hr_id, on='datetime', how='outer')
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


if __name__ == '__main__':
    COMBINED_DATA_PATH = "/Volumes/HP V236W/processed_dataset"
    SAVE_PATH = "/Users/kartikbhargava/coding/CE888/merged_data"

    if COMBINED_DATA_PATH != SAVE_PATH:
        os.mkdir(SAVE_PATH)
    print("Reading data ...")
    acc, eda, hr, temp = None, None, None, None
    signals = ['acc', 'eda', 'hr', 'temp']


    print("Starting...")
    pool = multiprocessing.Pool(len(signals))
    func = partial(read_parallel, COMBINED_DATA_PATH)
    results = pool.map(func, signals)
    pool.close()
    pool.join()
    print("Done :)")

    for i in results:
        globals()[i[0]] = i[1]

    # Merge data
    print('Merging Data ...')
    ids = eda['id'].unique()
    print(ids)
    columns = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'id', 'datetime']


    pool = multiprocessing.Pool(len(ids))
    func = partial(merge_parallel, acc, eda, hr, temp, columns)
    results = pool.map(func, ids)
    pool.close()
    pool.join()
    print("Merge Done")

    new_df = pd.concat(results, ignore_index=True)

    print("Saving data ...")
    new_df.to_csv(os.path.join(SAVE_PATH, "merged_data.csv"), index=False)
