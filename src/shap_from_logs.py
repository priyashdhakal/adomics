


import os
import shap 
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import accuracy_score

import arguments as args
import dataloader 
from model import get_model
import PlotUtils as plots

import json
import logging 
logging.getLogger().setLevel(logging.INFO)

logs_root_file = "logs"
PLOT_INDEX = 0

def get_hparams():
    weights_path_root = args.OptunaArguments.weights_path_root
    hparams_dict = json.load(open(os.path.join(weights_path_root, "best_params.json")))
    return hparams_dict
def load_model(dataset_dict, hparams_dict):
    model = get_model(dataset_dict, hparams_dict)
    logging.info("Loading weights from : " + args.OptunaArguments.weights_path_root)
    weights_path = os.path.join(args.OptunaArguments.weights_path_root, "best_weights.h5")
    model.load_weights(weights_path)
    return model

def get_shap_values(model, dataset_dict, biomarker_names_dict):
    key_values = list(dataset_dict.keys())
    print(f"Key Values: {key_values}")
    explainer = shap.DeepExplainer(model, 
                                     [dataset_dict[key_values[0]][1],
                                      dataset_dict[key_values[1]][1],
                                      dataset_dict[key_values[2]][1]])
    explainer.explainer.multi_input, explainer.explainer.multi_output = True, False
    shap_values = explainer.shap_values([dataset_dict[key_values[0]][1],
                                         dataset_dict[key_values[1]][1],
                                         dataset_dict[key_values[2]][1]], check_additivity=False)
    for i, feature_name in enumerate(biomarker_names_dict.keys()):
        plot = shap.summary_plot(shap_values[i], dataset_dict[key_values[i]][1], feature_names = biomarker_names_dict[feature_name], show = False)
        logging.info(f"Saving SHAP for dataset {key_values[i]} and index {PLOT_INDEX} and feature: {feature_name}")
        plt.savefig(os.path.join(args.PlotUtilArguments.plot_dir, f"plot_{PLOT_INDEX}_{feature_name}.png"))
        plt.close()
    return shap_values

def get_predictions(model, dataset_dict):
    val_x_arr = []
    for feature_name in args.DataSetArguments.feature_names:
        if feature_name not in dataset_dict:
            raise ValueError(f"Feature {feature_name} not found in dataset_dict")
        val_x_arr.append(dataset_dict[feature_name][1])
    val_y_arr = dataset_dict[args.DataSetArguments.feature_names[0]][3]
    if args.DataSetArguments.n_classes > 1:
        val_y_arr = keras.utils.to_categorical(val_y_arr, args.DataSetArguments.n_classes)
    y_pred = model.predict(val_x_arr)
    y_pred = np.where(y_pred > 0.5, 1, 0) 
    if args.DataSetArguments.n_classes > 1:
        val_y_arr = np.argmax(val_y_arr,axis=1)
        y_pred = np.argmax(y_pred,axis=1)
    acc = accuracy_score(val_y_arr, y_pred)
    logging.info(f"Accuracy: {acc}")
    plots.plot_confusion_matrix(val_y_arr, y_pred)

if __name__ == "__main__":
    dataset_dict, biomarker_names_dict  = dataloader.get_train_test_data()
    hparams_dict = get_hparams()
    model = load_model(dataset_dict, hparams_dict)
    get_predictions(model, dataset_dict)
    shap_values = get_shap_values(model, dataset_dict, biomarker_names_dict)

    print(shap_values[1].shape, len(shap_values))