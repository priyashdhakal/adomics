import arguments as args
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, GlobalMaxPool1D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import logging 
logging.getLogger().setLevel(logging.INFO)

def create_branch(input_layer):
    dense_methy = Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001))(input_layer)
    dense_methy = Dropout(0.5)(dense_methy)
    dense_methy = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001))(dense_methy)
    dense_methy = Dropout(0.3)(dense_methy)
    dense_methy = Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001))(dense_methy)
    dense_methy = Dropout(0.2)(dense_methy)
    return dense_methy

def get_model(dataset_dict, hparams_dict:dict):
    input_branch_arr = []
    feature_input_arr = []
    for feature_name in args.DataSetArguments.feature_names:
        if feature_name not in dataset_dict:
            raise ValueError(f"Feature {feature_name} not found in dataset_dict")
        feature_input = Input(shape=(dataset_dict[feature_name][0].shape[1],), name=feature_name)
        feature_input_arr.append(feature_input)
        logging.info(f"For Feature {feature_name} shape is {dataset_dict[feature_name][0].shape[1]}")
        feature_input_branch = create_branch(feature_input)
        input_branch_arr.append(feature_input_branch)
    merged = Concatenate()(input_branch_arr)
    merged_dense = Dense(hparams_dict["merged_1_size"], activation='relu', kernel_regularizer=regularizers.l1(0.001),name = "merged_1" )(merged)
    merged_dense = Dropout(hparams_dict["merged_1_dropout"])(merged_dense)
    merged_dense = Dense(hparams_dict["merged_2_size"], activation='relu', kernel_regularizer=regularizers.l1(0.001), name = "merged_2")(merged_dense)
    merged_dense = Dropout(hparams_dict["merged_2_dropout"])(merged_dense)
    merged_dense = Dense(hparams_dict["merged_3_size"], activation='relu', kernel_regularizer=regularizers.l1(0.001), name = "merged_3")(merged_dense)
    merged_dense = Dropout(hparams_dict["merged_3_dropout"])(merged_dense)
    output = Dense(
        args.DataSetArguments.n_classes, 
        activation=args.ModelArguments.last_layer_activation, 
        name = "merged_out"
    )(merged_dense)
    optimizer = args.ModelArguments.optimizer(learning_rate=hparams_dict["learning_rate"])
    model = Model(inputs=feature_input_arr, outputs=output)
    model.compile(optimizer=optimizer, loss=args.ModelArguments.loss, metrics=['accuracy'])
    logging.info("Model compiled")
    return model