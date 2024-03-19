import arguments as args 
import pandas as pd
import os
import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.preprocessing import StandardScaler
import tqdm
from tqdm import trange

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

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in trange(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

def preprocess_data(df):
    train_df, test_df, y_train, y_test = get_train_test(df)
    logging.info("finding correlation")
    corr_features = correlation(train_df, 0.85)
    logging.info(f'Correlated features: {len(corr_features)}')
    train_df.drop(corr_features, axis=1, inplace=True)
    test_df.drop(corr_features, axis=1, inplace=True)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, y_train, y_test

def get_train_test_data():
    df_dict = get_df()
    train_test_data = {}
    for feature, df in df_dict.items():
        train_scaled, test_scaled, y_train, y_test = preprocess_data(df)
        train_test_data[feature] = (train_scaled, test_scaled, y_train, y_test)
    return train_test_data
if __name__ == "__main__":
    df_dict = get_df()
    for feature, df in df_dict.items():
        train_scaled, test_scaled, y_train, y_test = preprocess_data(df)
        logging.info(f"Preprocessed {feature} data")
        logging.info(f"Train shape: {train_scaled.shape}")
        logging.info(f"Test shape: {test_scaled.shape}")
        logging.info(f"Train labels: {y_train.shape}")
        logging.info(f"Test labels: {y_test.shape}")
        logging.info("\n")