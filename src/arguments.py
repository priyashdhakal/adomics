from dataclasses import dataclass 
from tensorflow.keras.optimizers import Adam
import json 
import os 

allowed_data_types = [
    "BRCA",
    "KIRP",
    "KIRC",
    "LUSC",
    "ROSMAP",
]
@dataclass
class DataSetArguments:
    """
    Use this to select the dataset that you want to use: 
    For example: data_type = ROSMAP will use the rosmap dataset. 
    But take care of things like n_classes that should be written according to the outputs of the dataset. 
    For example, in the case of ROSMAP it is 1 (i.e binary classification)
    """
    data_type:str = "ROSMAP"
    if data_type not in allowed_data_types:
        raise ValueError(f"Data type must be one of {allowed_data_types}")
    n_classes:int = 1
    data_root:str = "../data"
    feature_names = ["methy", "mirna", "mrna"]

@dataclass
class ModelArguments:
    last_layer_activation:str = "sigmoid" if DataSetArguments.n_classes == 1 else "softmax"
    loss:str = "binary_crossentropy" if DataSetArguments.n_classes == 1 else "categorical_crossentropy"
    metrics:tuple = tuple(["accuracy"])
    optimizer = Adam

@dataclass
class TrainingArguments:
    train:bool = False
    epochs:int = 5
    batch_size:int = 32

    # callback arguments: 
    monitor:str = 'val_loss'
    patience:int = 50
    restore_best_weights:bool = True


@dataclass 
class OptunaArguments:
    use_optuna:bool = True 
    n_trials:int = 5
    feature_names_str = "_".join(DataSetArguments.feature_names)
    weights_path_root:str = f"logs/logs_{DataSetArguments.data_type}_{feature_names_str}"
    direction:str = "maximize"

class PlotUtilArguments:
    plot_status:bool = True
    plot_dir:str = os.path.join(OptunaArguments.weights_path_root, "plots")

# Export all dataclasses as json to weights_path_root
def export_as_json():
    with open(os.path.join(OptunaArguments.weights_path_root, "data_args.json"), "w") as f:
        json.dump(
            {
                "DataSetArguments": DataSetArguments().__dict__,
                "ModelArguments": ModelArguments().__dict__,
                "TrainingArguments": TrainingArguments().__dict__,
                "OptunaArguments": OptunaArguments().__dict__,
            },
            f,
            indent = 4
        )

