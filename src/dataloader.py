import random
random.seed(100)
import arguments as args 
import pandas as pd
import os
import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.preprocessing import StandardScaler
import tqdm
from tqdm import trange
import pickle
from collections import defaultdict

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

def get_train_test(df, k_fold = False, n_folds = 5, test_fold = 1):
    """
    TODO: If K-Fold Cross validation is required, it should take two more parameters: 
    1. k_fold(bool): use K-Fold or not 
    2. k(int): Number of folds
    If number of folds is given, this function should return a list of tuples of train and test data for that 
    fold. 
    Also, it should take into account that the randomness is maintained and is reproducible.
    """
    
    if not k_fold:
        train_df = df[df['Split']==1].drop('Split', axis=1)
        test_df = df[df['Split']==0].drop('Split', axis=1)
        y_train = train_df['Label']
        y_test = test_df['Label']
        return train_df.drop('Label', axis=1), test_df.drop('Label', axis=1), y_train, y_test
    if n_folds-1 <test_fold:
        raise ValueError("Test fold should be less than n_folds-1")
    df = df[df['Split']==1].drop('Split', axis=1)
    y = df['Label']
    len_ = len(df)
    indices = list(range(len_))
    random.shuffle(indices)
    fold_indices_dict = defaultdict(list)
    for fold_no in range(n_folds):
        for val in indices:
            if val%n_folds == fold_no:
                if fold_no == test_fold:
                    fold_indices_dict['val'].append(val)
                else:
                    fold_indices_dict['train'].append(val)

    train_indices = fold_indices_dict['train']
    val_indices = fold_indices_dict['val']
    train_df = df.iloc[train_indices]
    test_df = df.iloc[val_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[val_indices]
    return train_df.drop('Label', axis = 1), test_df.drop('Label', axis = 1), y_train, y_test, df.drop('Label', axis = 1) #last for getting correlation


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

def preprocess_data(df, feature_name, n_folds, fold_no):
    k_fold = False
    if n_folds > 1:
        k_fold = True 
    train_df, test_df, y_train, y_test, df = get_train_test(df, k_fold = k_fold, n_folds = n_folds, test_fold = fold_no)
    logging.info("finding correlation")
    corr_features = correlation(df, 0.85, feature_name)
    logging.info(f'Correlated features: {len(corr_features)}')
    train_df.drop(corr_features, axis=1, inplace=True)
    test_df.drop(corr_features, axis=1, inplace=True)
    biomarker_names_arr = test_df.columns.to_list()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, y_train, y_test, biomarker_names_arr

def get_train_test_data(n_folds, fold_no):
    df_dict = get_df()
    train_test_data = {}
    biomarker_names_dict = {}
    for feature, df in df_dict.items():
        train_scaled, test_scaled, y_train, y_test, biomarker_names_arr = preprocess_data(df, feature_name = feature, n_folds = n_folds, fold_no = fold_no)
        train_test_data[feature] = (train_scaled, test_scaled, y_train, y_test)
        biomarker_names_dict[feature] = biomarker_names_arr
    return train_test_data, biomarker_names_dict
if __name__ == "__main__":
    df_dict = get_df()
    for feature, df in df_dict.items():
        train_df, test_df, y_train, y_test = get_train_test(df, k_fold = True, n_folds = 5, test_fold = 4)
        print(len(list(train_df.columns)))
        print(train_df.iloc[1])