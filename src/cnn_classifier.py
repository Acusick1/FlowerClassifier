import os
import pathlib
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from datetime import datetime
from functools import partial
from typing import Any
from src.settings import MODEL_DIR, DATASET_DIR, IMG_SIZE, BATCH_SIZE, SEED

# Running options
DATASET_NAME = "Flowers"    # Name of dataset with the data directory
EVAL_ONLY = True           # Evaluate existing model or train new model
SAVE_MODEL = True           # Save the newly trained model
TUNE_MODEL = True           # Tune model hyperparameters or train a single model
MAX_EPOCHS = 100            # Maximum number of epochs per training cycle
TEST_SPLIT = 0.1            # Fraction of full dataset to be used for testing
VAL_SPLIT = 0.15            # Fraction of training dataset (after testing split) to be used for validation


def save_model(model: tf.keras.Sequential, name: str):
    """
    Saves model in consistent timestamped format
    :param model: Model to save
    :param name: Name of model (will be appended with timestamp)
    :return:
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_name = "_".join((name, timestamp))
    model.save(pathlib.Path(MODEL_DIR, save_name))


def get_model(name: str, latest: bool = False):
    """
    Retrieve model for model directory
    :param name: Name or pattern to search for within model directory
    :param latest: Whether to select the latest model (assumes saved using save_model function)
    :return: model
    """
    if not latest:
        # Load named model directly
        model_path = MODEL_DIR / name
    else:
        # Not sorted by default
        model_paths = list(sorted(MODEL_DIR.glob(name)))

        if not model_paths:
            raise FileNotFoundError(
                f"No model named '{name}' found in {MODEL_DIR}, please train a model first"
            )

        model_path = model_paths[-1]

    print(f"Loading model for prediction: {model_path}")
    model = tf.keras.models.load_model(model_path)

    return model


def get_model_config(hp: kt.HyperParameters()) -> dict[str, Any]:
    """
    Get model configuration parameters, returns values selected from each hyperparameter when called within a tuning
        loop, otherwise returns default values
    :param hp: HyperParameters object
    :return: Dictionary of model configuration parameters
    """
    hp.Int("conv_kernel_size", 3, 7, step=2, default=3),
    hp.Choice("layer1_units", [64, 128], default=128),
    hp.Choice("dropout", [0.25, 0.5], default=0.5),
    hp.Choice("learning_rate", [0.0005, 0.001, 0.005], default=0.001)
    config = hp.values

    return config


def tune_build_wrapper(hp: kt.HyperParameters(), *args, **kwargs) -> tf.keras.Sequential:
    """
    Wrapper around build_model function allow hyperparameter tuning, additional arguments passed directly to build_model
    :param hp: HyperParameters object
    :return: Compiled model
    """
    config = get_model_config(hp)
    model = build_model(config, *args, **kwargs)
    return model


def build_model(config: dict[str, Any], num_classes: int) -> tf.keras.Sequential:
    """
    Build CNN classifier with input configuration parameters and number of classes
    :param config: Model configuration parameters
    :param num_classes: Number of input classes
    :return: Compiled model
    """
    conv_layer = partial(tf.keras.layers.Conv2D,
                         kernel_size=config["conv_kernel_size"],
                         strides=1,
                         padding="SAME",
                         activation="relu")

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        conv_layer(filters=16),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        conv_layer(filters=32),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        conv_layer(filters=64),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        conv_layer(filters=128),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),  # Reshape layers to single array
        tf.keras.layers.Dense(config["layer1_units"], activation="relu"),  # Fully connected layer
        tf.keras.layers.Dropout(config["dropout"]),  # Dropout to avoid over fitting
        tf.keras.layers.Dense(num_classes),  # Fully connected layer with number of nodes = number of classes
        tf.keras.layers.Softmax()  # Normalise output to class probabilities
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["sparse_categorical_accuracy"])

    return model


def split_dataset(dataset: tf.data.Dataset, split: float = 0.8) -> (tf.data.Dataset, tf.data.Dataset):
    """
    Split full dataset into two partitions, defined by input fraction
    :param dataset: Full dataset
    :param split: Fraction to split dataset between 0 and 1
    :return: dataset 1 (split), dataset 2 (1 - split)
    """

    assert 0 < split < 1

    ds_size = len(dataset)
    left_size = int(split * ds_size)
    right_size = int((1 - split) * ds_size)

    left_ds = dataset.take(left_size)
    right_ds = dataset.skip(left_size).take(right_size)

    return left_ds, right_ds


def format_file_img(file_path: str):
    """
    Read and format an image from a file
    :param file_path: Path to file, specified as a string to allow use in tensorflow dataset mapping
    :return: formatted image
    """
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=IMG_SIZE[-1], expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE[:2])
    return img


def preprocess_dataset_img(file_path: str, class_names: np.array) -> (tf.Tensor, tf.int64):
    """
    Preprocess an image loaded from an on-disk dataset to be used in training/testing of a model, requires both input
        data and label to be returned
    :param file_path: Path to image file
    :param class_names: Numpy array of class names
    :return: formatted image, class label
    """

    def get_label():
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # File parent directory name is the class name
        one_hot = parts[-2] == class_names
        # Integer encode the label
        label = tf.argmax(one_hot)
        return label

    return format_file_img(file_path), get_label()


def configure_img_dataset(dataset: tf.data.Dataset, class_names: np.array, shuffle: bool = True) -> tf.data.Dataset:
    """
    Configure file based image dataset with loading/preprocessing functionality and performance
    :param dataset: File based image dataset
    :param class_names: Numpy array of class names
    :param shuffle: Option to randomise dataset, True by default
    :return: Configured dataset
    """
    dataset = dataset.map(lambda x: preprocess_dataset_img(x, class_names), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)

    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def train(full_set: tf.data.Dataset,
          class_names: np.array,
          val_split: float = 0.2,
          epochs: int = 10,
          tune: bool = False,
          model_name: str = "unnamed_model",
          ) -> tf.keras.Sequential:
    """
    Train a TensorFlow classifier based on input dataset
    :param full_set: Full training dataset
    :param class_names: Numpy array of class names
    :param val_split: Fraction to split full dataset to train and validation sets
    :param epochs: Maximum number of epochs per training cycle
    :param tune: Boolean to tune model hyperparameters
    :param model_name: Name of model (only necessary when tuning)
    :return: Fitted model
    """

    val_set, train_set = split_dataset(full_set, split=val_split)
    train_set = configure_img_dataset(train_set, class_names)
    val_set = configure_img_dataset(val_set, class_names)

    val_loss_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    if tune:
        tuner = kt.RandomSearch(
            partial(tune_build_wrapper, num_classes=len(class_names)),
            objective="val_loss",
            max_trials=10,
            directory=MODEL_DIR / "tuning_results",
            project_name=f"{model_name}")

        tuner.search(train_set, validation_data=val_set, epochs=epochs, callbacks=[val_loss_stop])
        model = tuner.get_best_models()[0]

        # Retraining on full dataset
        loss_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

        print("Retraining best model on full dataset:")
        full_set = configure_img_dataset(full_set, class_names)
        model.fit(full_set, epochs=epochs, callbacks=[loss_stop])
    else:
        config = get_model_config(kt.HyperParameters())
        model = build_model(config, len(class_names))
        model.fit(train_set, validation_data=val_set, epochs=epochs, callbacks=[val_loss_stop])

    return model


def get_file_dataset(dataset_path: pathlib.Path) -> tf.data.Dataset:
    """
    Get file list dataset from input path, assumes immediate subdirectories contain individual classes
    :param dataset_path: Path to dataset
    :return: combined dataset
    """
    print(f"Getting data from {dataset_path}")
    dataset = tf.data.Dataset.list_files(str(dataset_path.joinpath("*", "*")), shuffle=False)

    return dataset


def get_file_dataset_classes(dataset_path: pathlib.Path) -> np.array:
    """
    Get file list dataset from input path, assumes immediate subdirectories contain individual classes
    :param dataset_path: Path to dataset
    :return: names of each class (subdirectory names)
    """
    class_names = np.array(sorted([item.name for item in dataset_path.glob("*") if os.path.isdir(item)]))

    return class_names


def main():
    """
    Main running function to train/tune/evaluate a classifier
    """
    # Gather data
    dataset_path = DATASET_DIR / DATASET_NAME
    if not pathlib.Path.is_dir(dataset_path):
        raise NotADirectoryError(f"Dataset {DATASET_NAME} not found in {DATASET_DIR}")

    full_set = get_file_dataset(dataset_path)
    class_names = get_file_dataset_classes(dataset_path)

    # Shuffle to ensure varied train and test sets
    full_set = full_set.shuffle(len(full_set), seed=SEED)
    test_set, train_set = split_dataset(full_set, split=TEST_SPLIT)

    if EVAL_ONLY:

        model = get_model(f"{DATASET_NAME}*", latest=True)
    else:
        model = train(train_set,
                      class_names=class_names,
                      val_split=VAL_SPLIT,
                      epochs=MAX_EPOCHS,
                      tune=TUNE_MODEL,
                      model_name=DATASET_NAME)

        if SAVE_MODEL:
            save_model(model, DATASET_NAME)

    # Evaluate model
    eval_sets = {"Test set": test_set, "Full dataset": full_set}

    for name, dataset in eval_sets.items():
        dataset = configure_img_dataset(dataset, class_names, shuffle=False)

        label_idx = np.concatenate([l for _, l in dataset], axis=0)
        predictions = model.predict(dataset, verbose=0)
        confusion_matrix = tf.math.confusion_matrix(label_idx, tf.argmax(predictions, axis=1))
        print(f"{name} confusion matrix: \n {confusion_matrix}")

        _, test_acc = model.evaluate(dataset, verbose=0)
        print(f"{name} accuracy: {round(test_acc * 100, 2)}%")


if __name__ == "__main__":

    main()
