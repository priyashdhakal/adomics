


import os
import shap 
import numpy as np
import matplotlib.pyplot as plt

import arguments as args
import dataloader 
from model import get_model

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

def get_shap_values(model, dataset_dict):
    key_values = list(dataset_dict.keys())
    print(f"Key Values: {key_values}")
    explainer = shap.DeepExplainer(model, 
                                     [dataset_dict[key_values[0]][1],
                                      dataset_dict[key_values[1]][1],
                                      dataset_dict[key_values[2]][1]])
    explainer.explainer.multi_input, explainer.explainer.multi_output = True, False
    shap_values = explainer.shap_values([dataset_dict[key_values[0]][1],
                                         dataset_dict[key_values[1]][1],
                                         dataset_dict[key_values[2]][1]])
    plot = shap.summary_plot(shap_values[0], dataset_dict[key_values[0]][1], show = False)
    logging.info(f"Saving SHAP for dataset {key_values[0]} and index {PLOT_INDEX}")
    plt.savefig(os.path.join(args.PlotUtilArguments.plot_dir, f"plot_{PLOT_INDEX}.png"))
    return shap_values

if __name__ == "__main__":

    dataset_dict = dataloader.get_train_test_data()
    hparams_dict = get_hparams()
    model = load_model(dataset_dict, hparams_dict)
    shap_values = get_shap_values(model, dataset_dict)

    print(shap_values[1].shape, len(shap_values))