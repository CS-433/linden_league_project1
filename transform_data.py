import json
import os

import numpy as np

from helpers import load_csv_data


def map_columns(X, col_indices):
    """
    Maps specific values in the dataset to 0 or NaN based on a the BRFSS Codebook.

    Args:
        X (np.ndarray): input dataset
        col_indices (dict): mapping of column names to their respective indices in the dataset

    Returns:
        X (np.ndarray): modified dataset with values replaced by either 0 or NaN according to the format
    """

    X = X.copy()

    with open(os.path.join("format.json"), "r") as f:
        data_format = json.load(f)

    zero_values = ["None", "Not at any time", "Never", "No", "0"]
    missing_values = [
        "Refused",
        "Not Sure",
        "Don't know",
        "Missing",
        "Not asked",
    ]
    for col, idx in col_indices.items():
        if col not in data_format:
            continue

        for value, description in data_format[col].items():
            arr = X[:, idx]

            if description in zero_values:
                mask = np.isin(arr, [value])
                X[:, idx] = np.where(mask, 0, arr)
            elif description in missing_values:
                mask = np.isin(arr, [value])
                X[:, idx] = np.where(mask, np.nan, arr)

    return X


RAW_DATA_PATH = "data_raw"
TRANSFORMED_DATA_PATH = "data_transformed"


def main():
    x_train, x_test, y_train, train_ids, test_ids, col_indices = load_csv_data(
        RAW_DATA_PATH
    )
    print("Training set shape: ", x_train.shape)
    print("Test set shape: ", x_test.shape)

    x_train = map_columns(x_train, col_indices)
    x_test = map_columns(x_test, col_indices)

    print("Transformed training set shape: ", x_train.shape)
    print("Transformed test set shape: ", x_test.shape)

    if not os.path.exists(TRANSFORMED_DATA_PATH):
        os.makedirs(TRANSFORMED_DATA_PATH)

    # Save processed data
    for data, name in zip(
        [x_train, x_test, y_train, train_ids, test_ids],
        ["x_train", "x_test", "y_train", "train_ids", "test_ids"],
    ):
        np.save(os.path.join(TRANSFORMED_DATA_PATH, name + ".npy"), data)

    header = ", ".join([col for col in col_indices])

    with open(os.path.join(TRANSFORMED_DATA_PATH, "header.txt"), "w") as f:
        f.write(header)


if __name__ == "__main__":
    main()
