import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import tensorflow_addons as tfa
from visualizer import Plotter

# Set Seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
BASE_PATH = "data/ROSMAP/"
methy_path = os.path.join(BASE_PATH, "methy.csv")
mirna_path = os.path.join(BASE_PATH, "mirna.csv")
mrna_path = os.path.join(BASE_PATH, "mrna.csv")
if (
    not os.path.exists(methy_path)
    or not os.path.exists(mirna_path)
    or not os.path.exists(mrna_path)
):
    raise Exception("File not exists!")


def load_dataset():
    methy_df = pd.read_csv(methy_path, index_col=0)
    mirna_df = pd.read_csv(mirna_path, index_col=0)
    mrna_df = pd.read_csv(mrna_path, index_col=0)
    methy_df_wol = methy_df.drop("Label", axis=1)
    mirna_df_wol = mirna_df.drop("Label", axis=1)
    mrna_df_wol = mrna_df.drop("Label", axis=1)
    methy_df_wos = methy_df_wol.drop("Split", axis=1)
    mirna_df_wos = mirna_df_wol.drop("Split", axis=1)
    mrna_df_wos = mrna_df_wol.drop("Split", axis=1)

    combined_df = pd.concat([methy_df_wos, mirna_df_wos, mrna_df_wos], axis=1)
    combined_df["Label"] = methy_df["Label"]
    combined_df["Split"] = methy_df["Split"]
    return combined_df


def get_train_test(df):
    train_df = df[df["Split"] == 1].drop("Split", axis=1)
    test_df = df[df["Split"] == 0].drop("Split", axis=1)
    y_train = train_df.pop("Label")
    y_test = test_df.pop("Label")
    return train_df, test_df, y_train, y_test


def get_model(input_shape, trial):
    input_layer = Input(shape=input_shape)
    ######Parameters of the Model########
    n_dense1 = trial.suggest_int("n_dense1", 128, 256)
    dense_1_dropout = trial.suggest_float("dense_1_dropout", 0.1, 0.5)
    n_dense2 = trial.suggest_int("n_dense2", 64, 128)
    dense_2_dropout = trial.suggest_float("dense_2_dropout", 0.1, 0.5)
    n_dense3 = trial.suggest_int("n_dense3", 32, 64)
    dense_3_dropout = trial.suggest_float("dense_3_dropout", 0.1, 0.5)
    n_dense4 = trial.suggest_int("n_dense4", 16, 32)
    dense_4_dropout = trial.suggest_float("dense_4_dropout", 0.1, 0.5)
    ####Hyperparameters####
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    dense_layer1 = Dense(n_dense1, activation="relu", kernel_regularizer=l2(0.01))(
        input_layer
    )
    dropout_layer1 = Dropout(dense_1_dropout)(dense_layer1)
    dense_layer2 = Dense(n_dense2, activation="relu", kernel_regularizer=l2(0.01))(
        dropout_layer1
    )
    dropout_layer2 = Dropout(dense_2_dropout)(dense_layer2)
    dense_layer3 = Dense(n_dense3, activation="relu", kernel_regularizer=l2(0.01))(
        dropout_layer2
    )
    dropout_layer3 = Dropout(dense_3_dropout)(dense_layer3)
    dense_layer4 = Dense(n_dense4, activation="relu", kernel_regularizer=l2(0.01))(
        dropout_layer2
    )
    dropout_layer4 = Dropout(dense_4_dropout)(dense_layer4)

    output_layer = Dense(1, activation="sigmoid")(dropout_layer4)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def get_callbacks():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5",
        monitor="val_loss",
        save_best_only=True,
    )
    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    return [early_stopping]

def objective(trial, input_shape):
    model = get_model(input_shape, trial)
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test_scaled, y_test),
        callbacks=get_callbacks(),
    )
    return history.history["val_accuracy"][-1]

if __name__ == "__main__":
    combined_df = load_dataset()
    X_train, X_test, y_train, y_test = get_train_test(combined_df)
    min_max_scaler = StandardScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)
    input_shape = X_train_scaled.shape[1]
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, input_shape), n_trials=1000)
    print(study.best_trial)
    trials_df = study.trials_dataframe()
    # Export the dataframe to csv
    trials_df.to_csv("trials.csv")
    print("Bhayo")
