import arguments as args 
import dataloader 
import logging 
logging.getLogger().setLevel(logging.INFO)
from model import get_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 
import tensorflow as tf 
import keras
from sklearn.metrics import accuracy_score
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

def train_model(dataset_dict):
    model = get_model(dataset_dict)
    train_X_arr = [
        dataset_dict[args.DataSetArguments.feature_names[0]][0],
        dataset_dict[args.DataSetArguments.feature_names[1]][0],
        dataset_dict[args.DataSetArguments.feature_names[2]][0],
    ]
    val_x_arr = [
        dataset_dict[args.DataSetArguments.feature_names[0]][1],
        dataset_dict[args.DataSetArguments.feature_names[1]][1],
        dataset_dict[args.DataSetArguments.feature_names[2]][1],
    
    ]
    # Doesn't matter as all labels are same across modalities
    train_y_arr = dataset_dict[args.DataSetArguments.feature_names[0]][2]
    train_y_arr = keras.utils.to_categorical(train_y_arr, args.DataSetArguments.n_classes)
    val_y_arr = dataset_dict[args.DataSetArguments.feature_names[0]][3]
    val_y_arr = keras.utils.to_categorical(val_y_arr, args.DataSetArguments.n_classes)
    history = model.fit(
        train_X_arr, 
        train_y_arr, 
        epochs=args.TrainingArguments.epochs, 
        batch_size=args.TrainingArguments.batch_size, 
        validation_data=(val_x_arr, val_y_arr),
        callbacks=[EarlyStopping(monitor= 'val_loss', patience=50, restore_best_weights=True)]
    )
    y_pred = model.predict(val_x_arr)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    val_y_arr = np.argmax(val_y_arr,axis=1)
    y_pred = np.argmax(y_pred,axis=1)
    acc = accuracy_score(val_y_arr, y_pred)
    logging.info(f"Accuracy: {acc}")
    return acc
    
    
    
    
    
if __name__ == "__main__":
    dataset_dict = dataloader.get_train_test_data()
    for feature_name, (X_train, X_test, y_train, y_test) in dataset_dict.items():
        logging.info(f"Feature {feature_name} has shape {X_train.shape}")
    acc = train_model(dataset_dict)
    