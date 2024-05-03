import arguments as args 
import pandas as pd
import os
import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.preprocessing import StandardScaler
import tqdm
from tqdm import trange
import pickle

def get_df():
    base_path = args.DataSetArguments.data_root
    df_dict = {}
    for feature in args.DataSetArguments.feature_names:
        file_path = os.path.join(base_path, f"{args.DataSetArguments.data_type}/{feature}.csv")
        if not os.path.exists(file_path):
            raise Exception('File not found')
        logging.info(f"Reading {file_path}")
        df = pd.read_csv(file_path, index_col=0)
        df_dict[feature] = df
    return df_dict

def get_train_test(df):
    train_df = df[df['Split']==1].drop('Split', axis=1)
    test_df = df[df['Split']==0].drop('Split', axis=1)
    y_train = train_df['Label']
    y_test = test_df['Label']
    return train_df.drop('Label', axis=1), test_df.drop('Label', axis=1), y_train, y_test

def get_cache_filepath(feature_name):
    data_type = args.DataSetArguments.data_type
    cache_filename = f"{data_type}_{feature_name}_corr.pkl"
    cache_root = "/home/dhakal/MoBI/src/cache" #TODO: Fix this to get from  arguments.py
    cache_filepath = os.path.join(cache_root, cache_filename)
    return cache_filepath

def correlation(dataset, threshold, feature_name):
    cache_filepath = get_cache_filepath(feature_name)
    if os.path.exists(cache_filepath):
        logging.critical("Cache found, Loading from cache")
        with open(cache_filepath, "rb") as f:
            col_corr = pickle.load(f)
        return col_corr
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in trange(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    with open(cache_filepath, "wb") as f:
        logging.info(f"Saving cache to {cache_filepath}")
        pickle.dump(col_corr, f)
    return col_corr

def preprocess_data(df, feature_name):
    train_df, test_df, y_train, y_test = get_train_test(df)
    logging.info("finding correlation")
    corr_features = correlation(train_df, 0.85, feature_name)
    logging.info(f'Correlated features: {len(corr_features)}')
    train_df.drop(corr_features, axis=1, inplace=True)
    test_df.drop(corr_features, axis=1, inplace=True)
    biomarker_names_arr = test_df.columns.to_list()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, y_train, y_test, biomarker_names_arr

def get_train_test_data():
    df_dict = get_df()
    train_test_data = {}
    biomarker_names_dict = {}
    for feature, df in df_dict.items():
        train_scaled, test_scaled, y_train, y_test, biomarker_names_arr = preprocess_data(df, feature_name = feature)
        train_test_data[feature] = (train_scaled, test_scaled, y_train, y_test)
        biomarker_names_dict[feature] = biomarker_names_arr
    return train_test_data, biomarker_names_dict
if __name__ == "__main__":
    df_dict = get_df()
    for feature, df in df_dict.items():
        train_scaled, test_scaled, y_train, y_test = preprocess_data(df, feature_name = feature)
        logging.info(f"Preprocessed {feature} data")
        logging.info(f"Train shape: {train_scaled.shape}")
        logging.info(f"Test shape: {test_scaled.shape}")
        logging.info(f"Train labels: {y_train.shape}")
        logging.info(f"Test labels: {y_test.shape}")
        logging.info("\n")