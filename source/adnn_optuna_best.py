import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

def get_train_test(df):
    train_df = df[df['Split']==1].drop('Split', axis=1)
    test_df = df[df['Split']==0].drop('Split', axis=1)
    y_train = train_df['Label']
    y_test = test_df['Label']
    return train_df.drop('Label', axis=1), test_df.drop('Label', axis=1), y_train, y_test

# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


BASE_PATH = 'data/ROSMAP/'
methy_path = os.path.join(BASE_PATH, 'methy.csv')
mirna_path = os.path.join(BASE_PATH, 'mirna.csv')
mrna_path = os.path.join(BASE_PATH, 'mrna.csv')
if not os.path.exists(methy_path) or not os.path.exists(mirna_path) or not os.path.exists(mrna_path):
    raise Exception('File not found')

methy_df = pd.read_csv(methy_path, index_col=0)
mirna_df = pd.read_csv(mirna_path, index_col=0)
mrna_df = pd.read_csv(mrna_path, index_col=0)

methy_train_df, methy_test_df, methy_y_train, methy_y_test = get_train_test(methy_df)
mirna_train_df, mirna_test_df, mirna_y_train, mirna_y_test = get_train_test(mirna_df)
mrna_train_df, mrna_test_df, mrna_y_train, mrna_y_test = get_train_test(mrna_df)

methy_corr_features = correlation(methy_train_df, 0.85)
print('methy correlated features: ', len(methy_corr_features))
mirna_corr_features = correlation(mirna_train_df, 0.85)
print('mirna correlated features: ', len(mirna_corr_features))
mrna_corr_features = correlation(mrna_train_df, 0.85)
print('mrna correlated features: ', len(mrna_corr_features))

methy_train_df.drop(methy_corr_features, axis=1, inplace=True)
mirna_train_df.drop(mirna_corr_features, axis=1, inplace=True)
mrna_train_df.drop(mrna_corr_features, axis=1, inplace=True)

methy_test_df.drop(methy_corr_features, axis=1, inplace=True)
mirna_test_df.drop(mirna_corr_features, axis=1, inplace=True)
mrna_test_df.drop(mrna_corr_features, axis=1, inplace=True)

scaler_methy = StandardScaler()
scaler_mirna = StandardScaler()
scaler_mrna = StandardScaler()

methy_train_scaled = scaler_methy.fit_transform(methy_train_df)
methy_test_scaled = scaler_methy.transform(methy_test_df)

mirna_train_scaled = scaler_mirna.fit_transform(mirna_train_df)
mirna_test_scaled = scaler_mirna.transform(mirna_test_df)

mrna_train_scaled = scaler_mrna.fit_transform(mrna_train_df)
mrna_test_scaled = scaler_mrna.transform(mrna_test_df)

def create_branch(input_layer):
    dense_methy = Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001))(input_layer)
    dense_methy = Dropout(0.5)(dense_methy)
    dense_methy = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001))(dense_methy)
    dense_methy = Dropout(0.3)(dense_methy)
    dense_methy = Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001))(dense_methy)
    dense_methy = Dropout(0.2)(dense_methy)
    return dense_methy

def get_model():
    # merged_1_size = trial.suggest_int("merged_1_size", 64, 256)
    # merged_2_size = trial.suggest_int("merged_2_size", 32, 128)
    # merged_3_size = trial.suggest_int("merged_3_size", 16, 64)
    # merged_1_dropout = trial.suggest_float("merged_1_dropout", 0.1, 0.5)
    # merged_2_dropout = trial.suggest_float("merged_2_dropout", 0.1, 0.5)
    # merged_3_dropout = trial.suggest_float("merged_3_dropout", 0.1, 0.5)
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    learning_rate =  0.023718015 #0.01671121609413545
    merged_1_dropout = 0.438687619 #0.3081613502711167
    merged_1_size = 112 #215
    merged_2_dropout = 0.193897159 #0.18420741802575546
    merged_2_size = 49 #76
    merged_3_dropout = 0.154345902 #0.1453808705161605
    merged_3_size = 63 #64
    #869,869,0.8773584905660378,2024-01-30 19:50:40.909047,2024-01-30 19:50:52.016028,0 days 00:00:11.106981,0.01671121609413545,0.3081613502711167,215,0.18420741802575546,76,0.1453808705161605,64,COMPLETE


    input_methy = Input(shape=(methy_train_scaled.shape[1],), name='methy')
    input_mirna = Input(shape=(mirna_train_scaled.shape[1],), name='mirna')
    input_mrna = Input(shape=(mrna_train_scaled.shape[1],), name='mrna')
    methy_branch = create_branch(input_methy)
    mirna_branch = create_branch(input_mirna)
    mrna_branch = create_branch(input_mrna)
    merged = Concatenate()([methy_branch, mirna_branch, mrna_branch])
    merged_dense = Dense(merged_1_size, activation='relu', kernel_regularizer=regularizers.l1(0.001),name = "merged_1" )(merged)
    merged_dense = Dropout(merged_1_dropout)(merged_dense)
    merged_dense = Dense(merged_2_size, activation='relu', kernel_regularizer=regularizers.l1(0.001), name = "merged_2")(merged_dense)
    merged_dense = Dropout(merged_2_dropout)(merged_dense)
    merged_dense = Dense(merged_3_size, activation='relu', kernel_regularizer=regularizers.l1(0.001), name = "merged_3")(merged_dense)
    merged_dense = Dropout(merged_3_dropout)(merged_dense)
    # merged_dense = Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.001))(merged_dense)
    # merged_dense = Dropout(0.2)(merged_dense)

    output = Dense(1, activation='sigmoid', name = "merged_out")(merged_dense)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = Model(inputs=[input_methy, input_mirna, input_mrna], outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def objective():
    model = get_model()
    history = model.fit([methy_train_scaled, mirna_train_scaled, mrna_train_scaled], 
          methy_y_train, 
          epochs=500, 
          batch_size=32, 
          validation_data=([methy_test_scaled, mirna_test_scaled, mrna_test_scaled], methy_y_test),
          callbacks=[EarlyStopping(monitor= 'val_loss', patience=50, restore_best_weights=True)])
    y_pred = model.predict([methy_test_scaled, mirna_test_scaled, mrna_test_scaled])
    y_pred = np.where(y_pred > 0.5, 1, 0)
    acc = accuracy_score(methy_y_test, y_pred)
    return acc


if __name__ == "__main__":
    print(objective())