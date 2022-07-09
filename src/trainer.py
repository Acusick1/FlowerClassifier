import os
import pathlib
import argparse
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from datetime import datetime
from src.settings import MODEL_DIR, DATASET_DIR, IMG_SIZE, BATCH_SIZE, SEED


def get_model_config(hp: kt.HyperParameters()):

    hp.Int("conv_kernel_size", 3, 7, step=2, default=3),
    hp.Choice("layer1_units", [64, 128], default=128),
    hp.Choice("dropout", [0.25, 0.5], default=0.5),
    hp.Choice("learning_rate", [0.0005, 0.001, 0.005], default=0.001)
    config = hp.values

    return config


def tune_wrapper(hp: kt.HyperParameters()):

    # TODO: Hardcoding
    config = get_model_config(hp)
    model = create_model(config, 17)
    return model


def create_model(config, num_classes: int) -> tf.keras.Sequential:

    conv_layer = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=config["conv_kernel_size"],
            strides=1,
            padding="SAME",
            activation="relu")
    pool_layer = tf.keras.layers.MaxPooling2D(pool_size=2)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        conv_layer,
        pool_layer,
        conv_layer,
        pool_layer,
        conv_layer,
        pool_layer,
        conv_layer,
        pool_layer,
        tf.keras.layers.Flatten(),  # Reshape layers to single array
        tf.keras.layers.Dense(config["layer1_units"], activation="relu"),  # Fully connected layer
        tf.keras.layers.Dropout(config["dropout"]),  # 50% dropout to avoid over fitting
        tf.keras.layers.Dense(num_classes),  # Fully connected layer with number of nodes = number of classes
        tf.keras.layers.Softmax()  # Normalise output to class probabilities
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model


def split_dataset(dataset: tf.data.Dataset, split: float = 0.8) -> (tf.data.Dataset, tf.data.Dataset):
    """
    Split full dataset by
    :param dataset: Full dataset
    :param split: Fraction to split dataset between 0 and 1
    :return: split dataset 1, split dataset 2
    """

    assert 0 < split < 1

    ds_size = len(dataset)
    left_size = int(split * ds_size)
    right_size = int((1 - split) * ds_size)

    left_ds = dataset.take(left_size)
    right_ds = dataset.skip(left_size).take(right_size)

    return left_ds, right_ds


def partition_ds(dataset: tf.data.Dataset, train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1):
    """
    Split full dataset into train, validate, test subsets
    :param dataset: Full dataset
    :param train_split: Fraction of full dataset to use for training
    :param val_split: Fraction of full dataset to use for validation
    :param test_split: Fraction of full dataset to use for testing
    :return: train, validation, test datasets
    """
    assert (train_split + test_split + val_split) == 1

    ds_size = len(dataset)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def preprocess_img_path(file_path, class_names: np.array) -> (tf.Tensor, tf.int64):
    """
    Preprocess an image loaded from an on-disk dataset
    :param file_path: Path to image file
    :param class_names: Dataset class names
    :return: formatted image, class label
    """

    def format_img():
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img, channels=IMG_SIZE[-1], expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE[:2])
        return img

    def get_label():
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        label = tf.argmax(one_hot)
        return label

    return format_img(), get_label()


def configure_dataset_performance(dataset: tf.data.Dataset) -> tf.data.Dataset:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def train(dataset_path: pathlib.Path, epochs: int = 10, tune=True) -> None:

    print(f"Getting data from {dataset_path}")
    ds, class_names = get_data(dataset_path)

    full_train_ds, test_ds = split_dataset(ds, split=0.9)
    train_ds, val_ds = split_dataset(full_train_ds, split=0.85)

    train_ds = train_ds.map(lambda x: preprocess_img_path(x, class_names), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x: preprocess_img_path(x, class_names), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x: preprocess_img_path(x, class_names), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = configure_dataset_performance(train_ds)
    val_ds = configure_dataset_performance(val_ds)
    test_ds = configure_dataset_performance(test_ds)
    val_loss_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    if tune:
        tuner = kt.RandomSearch(
            tune_wrapper,
            objective="val_loss",
            max_trials=10,
            directory=MODEL_DIR,
            project_name=f"{dataset_path.name}_tuned")

        tuner.search(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[val_loss_stop])
        model = tuner.get_best_models()[0]

        # Retraining on full dataset
        loss_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

        full_train_ds = full_train_ds.map(lambda x: preprocess_img_path(x, class_names),
                                          num_parallel_calls=tf.data.AUTOTUNE)

        print("Retraining best model on full dataset")
        full_train_ds = configure_dataset_performance(full_train_ds)
        model.fit(full_train_ds, epochs=epochs, callbacks=[loss_stop])
    else:
        config = get_model_config(kt.HyperParameters())
        model = create_model(config, len(class_names))
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[val_loss_stop])

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"\nTest accuracy: {round(test_acc * 100, 2)}%")

    # Saving
    pathlib.Path.mkdir(pathlib.Path(MODEL_DIR), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "_".join((dataset_path.name, timestamp))
    model.save(pathlib.Path(MODEL_DIR, model_name))


def get_data(dataset_path: pathlib.Path):

    list_ds = tf.data.Dataset.list_files(str(dataset_path.joinpath("*", "*")), shuffle=False)
    list_ds = list_ds.shuffle(len(list_ds), seed=SEED)

    class_names = np.array(sorted([item.name for item in dataset_path.glob("*") if os.path.isdir(item)]))

    return list_ds, class_names


def main(argv=None):

    parser = argparse.ArgumentParser(description="Training a CNN based on specified dataset")

    parser.add_argument("--dataset",
                        default="Flowers",
                        type=str,
                        help=f"Specify dataset within dataset directory ({DATASET_DIR})",
                        )

    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help=f"Number of training cycles",
                        )

    args = parser.parse_args(argv)

    dataset_path = pathlib.Path(DATASET_DIR, args.dataset)
    if not pathlib.Path.is_dir(dataset_path):
        raise NotADirectoryError(f"Dataset {args.dataset} not found in {DATASET_DIR}")

    train(dataset_path, epochs=args.epochs)


if __name__ == "__main__":

    main()
