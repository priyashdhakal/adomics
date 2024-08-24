seed = 123
import numpy as np 
import tensorflow as tf 
import keras
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)
import arguments as args 
import dataloader 
import logging 
logging.getLogger().setLevel(logging.INFO)
from model import get_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
import os

import optuna
import json




BEST_ACCURACY = 0

def get_callbacks():
    early_stopping_callback = EarlyStopping(monitor= args.TrainingArguments.monitor,
                                 patience=args.TrainingArguments.patience,
                                 restore_best_weights=args.TrainingArguments.restore_best_weights)
    return [early_stopping_callback]

def train_model(dataset_dict, hparams_dict):
    import PlotUtils as plots
    model = get_model(dataset_dict, hparams_dict)
    train_X_arr = []
    for feature_name in args.DataSetArguments.feature_names:
        if feature_name not in dataset_dict:
            raise ValueError(f"Feature {feature_name} not found in dataset_dict")
        train_X_arr.append(dataset_dict[feature_name][0])
    val_x_arr = []
    for feature_name in args.DataSetArguments.feature_names:
        if feature_name not in dataset_dict:
            raise ValueError(f"Feature {feature_name} not found in dataset_dict")
        val_x_arr.append(dataset_dict[feature_name][1])
    # Doesn't matter as all labels are same across modalities
    train_y_arr = dataset_dict[args.DataSetArguments.feature_names[0]][2]
    val_y_arr = dataset_dict[args.DataSetArguments.feature_names[0]][3]
    if args.DataSetArguments.n_classes > 1:
        train_y_arr = keras.utils.to_categorical(train_y_arr, args.DataSetArguments.n_classes)
        val_y_arr = keras.utils.to_categorical(val_y_arr, args.DataSetArguments.n_classes)
    if args.TrainingArguments.train:
        history = model.fit(
            train_X_arr, 
            train_y_arr, 
            epochs=args.TrainingArguments.epochs, 
            batch_size=args.TrainingArguments.batch_size, 
            validation_data=(val_x_arr, val_y_arr),
            callbacks= get_callbacks()
        )
    else:
        logging.info("Skipping training, Loading Pretrained Weights")
        weights_path = os.path.join(args.OptunaArguments.weights_path_root, "best_weights.h5")
        model.load_weights(weights_path)
    y_pred = model.predict(val_x_arr)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    if args.DataSetArguments.n_classes > 1:
        val_y_arr = np.argmax(val_y_arr,axis=1)
        y_pred = np.argmax(y_pred,axis=1)
    acc = accuracy_score(val_y_arr, y_pred)
    if args.PlotUtilArguments.plot_status and not args.TrainingArguments.train:
        plots.plot_confusion_matrix(val_y_arr, y_pred)
    if not args.TrainingArguments.train:
        return acc
    global BEST_ACCURACY
    if acc > BEST_ACCURACY:
        if args.PlotUtilArguments.plot_status:
            logging.info(f"Plotting History for best model")
            plots.plot_history(history)
        weights_path = os.path.join(args.OptunaArguments.weights_path_root, "best_weights.h5")
        model.save(weights_path) #TODO: replace this with proper name
        best_params_filepath = os.path.join(args.OptunaArguments.weights_path_root, "best_params.json")
        with open(best_params_filepath, "w") as f:
            json.dump(hparams_dict, f)
            f.write(f"accuracy: {acc}")
        BEST_ACCURACY = acc
    logging.info(f"Accuracy: {acc}")
    return acc
    
    
def objective(trial):
    hparams_dict = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "merged_1_dropout": trial.suggest_float("merged_1_dropout", 0.1, 0.5),
        "merged_1_size": trial.suggest_int("merged_1_size", 64, 256),
        "merged_2_dropout": trial.suggest_float("merged_2_dropout", 0.1, 0.5),
        "merged_2_size": trial.suggest_int("merged_2_size", 32, 128),
        "merged_3_dropout": trial.suggest_float("merged_3_dropout", 0.1, 0.5),
        "merged_3_size": trial.suggest_int("merged_3_size", 16, 64),
    }
    acc = train_model(dataset_dict, hparams_dict)
    params_filepath = os.path.join(args.OptunaArguments.weights_path_root, "params.txt")
    with open(params_filepath, "a") as f:
        f.write(str(trial.params))
        f.write(f"accuracy: {acc}")
        f.write("\n")
    return acc
    
    
if __name__ == "__main__":
    n_folds = 5
    for fold_no in range(n_folds):
        ########################################################
        args.OptunaArguments.weights_path_root = f"{args.OptunaArguments.weights_path_root}/{fold_no}"
        args.PlotUtilArguments.plot_dir =  os.path.join(args.OptunaArguments.weights_path_root, "plots")
        print(f"Plot Dir is: {args.PlotUtilArguments.plot_dir}")
        raise SystemExit
        if not os.path.exists(args.OptunaArguments.weights_path_root):
            logging.info(f"Creating directory {args.OptunaArguments.weights_path_root}")
            os.makedirs(args.OptunaArguments.weights_path_root)
        else:
            logging.critical(f"Directory {args.OptunaArguments.weights_path_root} already exists, Files can be replaced")
        ######################################################################
        dataset_dict, _  = dataloader.get_train_test_data(n_folds, fold_no)
        for feature_name, (X_train, X_test, y_train, y_test) in dataset_dict.items():
            logging.info(f"Feature {feature_name} has shape {X_train.shape}")
        if args.OptunaArguments.use_optuna and args.TrainingArguments.train:
            study = optuna.create_study(direction=args.OptunaArguments.direction)
            study.optimize(objective, n_trials=args.OptunaArguments.n_trials)
            logging.info(f"Best trial: {study.best_trial}")
            logging.info(f"Saving Parameters")
            args.export_as_json()
        else:
            best_params_filepath = os.path.join(args.OptunaArguments.weights_path_root, f"{fold_no}_best_params.json")
            if not os.path.exists(best_params_filepath):
                raise ValueError(f"File {best_params_filepath} not found")
            with open(best_params_filepath, "r") as f:
                best_params = json.load(f)
            
            acc = train_model(dataset_dict, best_params)
            logging.info(f"Accuracy: {acc}")
    