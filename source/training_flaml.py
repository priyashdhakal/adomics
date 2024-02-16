import pandas as pd 
import numpy as np 
from flaml import AutoML
import os


def get_train_test(df):
    train_df = df[df["Split"]==1].drop("Split", axis=1)
    test_df = df[df["Split"]==0].drop("Split", axis=1)
    y_train = train_df.pop("Label")
    y_test = test_df.pop("Label")
    return (train_df.values, y_train.values.astype("int32")), (test_df.values, y_test.values.astype("int32"))

def automl_predict(train_data, test_data, modality_type, time_budget = int(4*60*60)):
    automl = AutoML()

    automl_settings = {
    "time_budget": time_budget,  # in seconds
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": f"automl_logs/{modality_type}.log",
    "seed":1000
    }
    automl.fit(X_train=train_data[0], y_train=train_data[1],
        **automl_settings)
    predictions = automl.predict(test_data[0])
    accuracy = np.where(predictions == test_data[1], 1, 0).sum()/len(predictions)
    print(f"Accuracy for Modality: {modality_type} is {accuracy}")
    return predictions

def ensemble_predictions(predictions_arr):
    predictions = np.mean(predictions_arr, axis=0)
    predictions = np.where(predictions > 0.5, 1, 0)
    return predictions


if __name__ == "__main__":
    base_path = "data/ROSMAP"
    methy_path = os.path.join(base_path, "methy.csv")
    mirna_path = os.path.join(base_path, "mirna.csv")
    mrna_path = os.path.join(base_path, "mrna.csv")
    if not os.path.exists(methy_path) or not os.path.exists(mirna_path) or not os.path.exists(mrna_path):
        raise ValueError("Data Not found")

    methy_df = pd.read_csv(methy_path, index_col=0)
    mirna_df = pd.read_csv(mirna_path, index_col=0)
    mrna_df = pd.read_csv(mrna_path, index_col=0)
    train_data_methy, test_data_methy = get_train_test(methy_df)
    train_data_mirna, test_data_mirna = get_train_test(mirna_df)
    train_data_mrna, test_data_mrna = get_train_test(mrna_df)
    predictions_arr = []
    methy_predictions = automl_predict(train_data_methy, test_data_methy, "methy")
    predictions_arr.append(methy_predictions)
    mirna_predictions = automl_predict(train_data_mirna, test_data_mirna, "mirna")
    predictions_arr.append(mirna_predictions)
    mrna_predictions = automl_predict(train_data_mrna, test_data_mrna, "mrna")
    predictions_arr.append(mrna_predictions)
    predictions_arr = np.array(predictions_arr)
    ensemble_predictions = ensemble_predictions(predictions_arr)
    print(np.where(ensemble_predictions == test_data_methy[1], 1, 0).sum()/len(ensemble_predictions))
