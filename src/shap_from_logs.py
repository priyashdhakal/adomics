


import os
import shap 
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc

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
    logging.info("Loading hparams from : " + os.path.join(weights_path_root, "best_params.json"))
    hparams_dict = json.load(open(os.path.join(weights_path_root, "best_params.json")))
    return hparams_dict
def load_model(dataset_dict, hparams_dict):
    model = get_model(dataset_dict, hparams_dict)
    logging.info("Loading weights from : " + args.OptunaArguments.weights_path_root)
    weights_path = os.path.join(args.OptunaArguments.weights_path_root, "best_weights.h5")
    model.load_weights(weights_path)
    return model

def get_shap_values(model, dataset_dict, biomarker_names_dict, curr_class = None):
    if curr_class is not None:
        to_add = f"class_{curr_class}"
    else:
        to_add = ""
    key_values = list(dataset_dict.keys())
    print(f"Key Values: {key_values}")
    # explainer = shap.DeepExplainer(model, 
    #                                  [dataset_dict[key_values[0]][1],
    #                                   dataset_dict[key_values[1]][1],
    #                                   dataset_dict[key_values[2]][1]])
    dataset_arr_dict = []
    key_values_idx = 1 # 0 = train data, so ignore
    for i, _ in enumerate(biomarker_names_dict.keys()):
        dataset_arr_dict.append(dataset_dict[key_values[i]][key_values_idx])

    explainer = shap.DeepExplainer(model, dataset_arr_dict)
    explainer.explainer.multi_input, explainer.explainer.multi_output = True, False
    # shap_values = explainer.shap_values([dataset_dict[key_values[0]][1],
    #                                      dataset_dict[key_values[1]][1],
    #                                      dataset_dict[key_values[2]][1]], check_additivity=False)

    shap_values = explainer.shap_values(dataset_arr_dict, check_additivity=False)
    for i, feature_name in enumerate(biomarker_names_dict.keys()):
        #IMP: Change returning feature names in line 963 of _beeswarm.py to return feature names
        # else, it will return nothing
        feature_names = shap.summary_plot(shap_values[i], dataset_dict[key_values[i]][key_values_idx], feature_names = biomarker_names_dict[feature_name], show = False, show_values_in_legend = True)
        logging.info(f"Saving Feature Names: ")
        feature_names_save_path = os.path.join(args.PlotUtilArguments.plot_dir, f"feature_names_{PLOT_INDEX}_{feature_name}.txt")
        feature_names = reversed(feature_names)
        with open(feature_names_save_path, "w") as f:
            f.write("\n".join(feature_names))
        logging.info(f"Saving SHAP for dataset {key_values[i]} and index {PLOT_INDEX} and feature: {feature_name} and class: {to_add}")
        plt.savefig(os.path.join(args.PlotUtilArguments.plot_dir, f"plot_{PLOT_INDEX}_{feature_name}_{to_add}.png"))
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
    # Draw ROC Curve 
    plots.plot_roc_curve(val_y_arr, y_pred)
    # Finish drawing ROC Curve
    y_pred = np.where(y_pred > 0.5, 1, 0) 
    if args.DataSetArguments.n_classes > 1:
        val_y_arr = np.argmax(val_y_arr,axis=1)
        y_pred = np.argmax(y_pred,axis=1)
    # acc = accuracy_score(val_y_arr, y_pred)
    # precision = precision_score(val_y_arr, y_pred)
    # recall = recall_score(val_y_arr, y_pred)
    # f1 = f1_score(val_y_arr, y_pred)
    # with open(os.path.join(args.PlotUtilArguments.plot_dir, "metrics.txt"), "w") as f:
    #     f.write(f"Accuracy: {acc}\n")
    #     f.write(f"Precision: {precision}\n")
    #     f.write(f"Recall: {recall}\n")
    #     f.write(f"F1: {f1}\n")
    # logging.info(f"Accuracy: {acc}")
    plots.plot_confusion_matrix(val_y_arr, y_pred)


def separate_dataset_dict(dataset_dict, n_classes = 5):
    """
    dataset_dict: 
        - methy
        - mirna
        - mrna //these for all methy, mirna and mrna
          -train
          -test
          -train_label
          -test_label
    
    Need to separate this dataset into:
    dataset_lbl_1_dict, dataset_lbl_2_dict, ..., dataset_lbl_n_dict


    """
    dataset_dict_arr = []
    for i in range(n_classes):
        curr_dataset_dict = {}
        for dataset_name in dataset_dict:
            curr_dataset_dict[dataset_name] = []
            all_train_labels = dataset_dict[dataset_name][2].values
            all_train_data = dataset_dict[dataset_name][0]
            all_test_labels = dataset_dict[dataset_name][3].values
            all_test_data = dataset_dict[dataset_name][1]
            # get indices of all_train_labels where label is i
            train_indices = np.where(all_train_labels == i)
            test_indices = np.where(all_test_labels == i)
            curr_dataset_dict[dataset_name] = [all_train_data[train_indices],
                                               all_test_data[test_indices],
                                               all_train_labels[train_indices],
                                               all_test_labels[test_indices]]
        
        dataset_dict_arr.append(curr_dataset_dict)
    return dataset_dict_arr

    


if __name__ == "__main__":
    dataset_dict, biomarker_names_dict  = dataloader.get_train_test_data()
    hparams_dict = get_hparams()
    if args.DataSetArguments.data_type == "BRCA":
        dataset_dict_arr = separate_dataset_dict(dataset_dict, args.DataSetArguments.n_classes)
        for i, curr_dataset_dict in enumerate(dataset_dict_arr):
            model = load_model(curr_dataset_dict, hparams_dict)
            get_predictions(model, curr_dataset_dict)
            shap_values = get_shap_values(model, curr_dataset_dict, biomarker_names_dict, i)
            print(shap_values[0].shape, len(shap_values))
    else:
        model = load_model(dataset_dict, hparams_dict)
        get_predictions(model, dataset_dict)
        shap_values = get_shap_values(model, dataset_dict, biomarker_names_dict)

        print(shap_values[0].shape, len(shap_values))