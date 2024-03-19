import pandas as pd 
import os 
import numpy as np 
from random import shuffle 
import logging 
logging.basicConfig(level=logging.INFO)

dataset_types = [
    "mrna", 
    "mirna", 
    "methy"
]

def get_combined_csv(data_df, labels_df):
    df = pd.concat([labels_df, data_df], axis=1)
    curr_col_names = list(df.columns)
    curr_col_names[0] = 'Label'
    df.columns = curr_col_names
    train_splits = [1 for i in range(int(0.8*len(df)))] + [0 for i in range(int(0.2*len(df)))]
    if len(train_splits) < len(df):
        logging.critical(f"Length of train_splits is less than length of df. Length of train_splits: {len(train_splits)}, Length of df: {len(df)}")
        train_splits += [0]
    elif len(train_splits) > len(df):
        logging.critical(f"Length of train_splits is greater than length of df. Length of train_splits: {len(train_splits)}, Length of df: {len(df)}")
        train_splits = train_splits[:len(df)]
    shuffle(train_splits)
    df['Split'] = train_splits 
    return df

def convert_data(data_filepath):
    dataset_name = data_filepath.split("/")[-1]
    for data_type in dataset_types:
        data_path = f"{data_filepath}/{dataset_name}_{data_type}.txt"
        labels_path = f"{data_filepath}/{dataset_name}_label.txt"
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise ValueError(f"Either {data_path} or {labels_path} does not exist")
        logging.info(f"Reading data from {data_path}")
        logging.info(f"Reading labels from {labels_path}")
        data_df = pd.read_csv(data_path, sep="\t")
        labels_df = pd.read_csv(labels_path, sep="\t")
        combined_df = get_combined_csv(data_df, labels_df)
        print(combined_df.head())
        combined_filepath = os.path.join(data_filepath, f"{data_type}.csv")
        logging.info(f"Writing combined data to {combined_filepath}")
        combined_df.to_csv(combined_filepath)

if __name__ == "__main__":
    data_filepath = "/home/dhakal/MoBI/data/KIRC"
    convert_data(data_filepath)