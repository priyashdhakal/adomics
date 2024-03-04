from dataclasses import dataclass 
from tensorflow.keras.optimizers import Adam

allowed_data_types = [
    "BRCA",
    "KIRP",
    "KIRC",
    "LUSC",
    "ROSMAP",
]
@dataclass
class DataSetArguments:
    data_type:str = "BRCA"
    if data_type not in allowed_data_types:
        raise ValueError(f"Data type must be one of {allowed_data_types}")
    n_classes:int = 5
    data_root:str = "../data"
    feature_names = ["methy", "mirna", "mrna"]

@dataclass
class ModelArguments:
    last_layer_activation = "sigmoid" if DataSetArguments.n_classes == 1 else "softmax"
    loss = "binary_crossentropy" if DataSetArguments.n_classes == 1 else "categorical_crossentropy"
    metrics = ["accuracy"]
    optimizer = Adam

@dataclass
class TrainingArguments:
    epochs = 500
    batch_size = 32
