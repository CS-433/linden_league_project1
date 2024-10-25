import os

import numpy as np

from columns import (
    get_all_binary_categorical_columns,
    get_all_columns,
    get_random_column_subset,
    get_selected_binary_categorical_columns,
    get_selected_columns,
)
from helpers import load_csv_data
from parser_helpers import create_preprocessing_parser
from transform_data import map_columns


def compute_mode(arr):
    """
    Computes the mode of an array, ignoring NaN values.

    Args:
        arr (np.ndarray): input array

    Returns:
        float or np.nan: mode of the array or NaN if array is empty.
    """
    values, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
    if len(counts) == 0:
        return np.nan
    max_count_index = np.argmax(counts)
    return values[max_count_index]


def transform_binary_columns(X, col_indices, columns):
    """
    Transforms binary categorical columns in X.

    Args:
        X (np.ndarray): data
        col_indices (dict): mapping of column names to indices for data
        columns (list): binary categorical column names
    """
    for col in columns:
        idx = col_indices[col]
        arr = X[:, idx]
        mask = np.isin(arr, [1, 2])
        X[:, idx] = np.where(mask, arr - 1, np.nan)


def transform_columns(X, col_indices, binary_categorical_columns):
    """
    Transforms columns in the dataset X according to specified rules.
    Rules are based on answers in the questionnaire used for generation of the dataset.
    See https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf to better understand the transformations.

    Args:
        X (np.ndarray): data
        col_indices (dict): mapping of column names to indices for data
        binary_categorical_columns (list): list of columns names for the binary categorical features
    Returns:
        X (np.ndarray): transformed data array
    """
    # Copy X to avoid modifying original data
    X = map_columns(X, col_indices)

    transform_binary_columns(X, col_indices, binary_categorical_columns)

    return X


def process_dataset(
    X,
    stats,
    numerical_columns,
    categorical_columns,
    multi_class_categorical_columns,
    col_indices,
):
    """
    Processes the dataset X by imputing missing values, standardizing numerical columns,
    and one-hot encoding categorical columns.

    Args:
        X (np.ndarray): data
        stats (dict): statistics for imputation and scaling
        numerical_columns (list): numerical column names
        categorical_columns (list): categorical column names
        multi_class_categorical_columns (list): multi-class categorical columns
        col_indices (dict): mapping of column names to indices for data

    Returns:
        X_array (np.ndarray): processed data
        new_col_indices (dict): mapping of column names to indices for processed data
    """
    X_new = []
    new_col_indices = {}
    col_idx = 0

    # Process numerical columns
    for col in numerical_columns:
        idx = col_indices[col]
        arr = X[:, idx]
        # Impute missing values with mean
        arr = np.where(np.isnan(arr), stats[col]["mean"], arr)
        # Standardize
        arr = (arr - stats[col]["mean"]) / stats[col]["std"]
        X_new.append(arr.reshape(-1, 1))
        new_col_indices[col] = col_idx
        col_idx += 1

    # Process categorical columns
    for col in categorical_columns:
        idx = col_indices[col]
        arr = X[:, idx]
        # Impute missing values with mode
        arr = np.where(np.isnan(arr), stats[col]["mode"], arr)

        if col in multi_class_categorical_columns:
            unique_values = stats[col]["unique_values"]
            # One-hot encode
            for val in unique_values:
                one_hot_arr = (arr == val).astype(float).reshape(-1, 1)
                X_new.append(one_hot_arr)
                new_col_name = f"{col}_{val}"
                new_col_indices[new_col_name] = col_idx
                col_idx += 1
        else:
            X_new.append(arr.reshape(-1, 1))
            new_col_indices[col] = col_idx
            col_idx += 1

    # Stack the columns to form a 2D array
    X_array = np.hstack(X_new)

    return X_array, new_col_indices


def clean_data(
    X_train,
    X_test,
    col_indices,
    numerical_columns,
    categorical_columns,
    binary_categorical_columns,
):
    """
    Cleans and processes the training and test datasets.

    Args:
        X_train (np.ndarray): training data
        X_test (np.ndarray): test data.
        col_indices (dict): mapping of column names to their indices in the data

    Returns:
        X_train (np.array): cleaned training data
        X_test (np.array): cleaned test data
        columns (dict): mapping of column name to idx for cleaned data
    """
    multi_class_categorical_columns = [
        col
        for col in categorical_columns
        if col not in binary_categorical_columns
    ]
    stats = {}

    X_train = transform_columns(
        X_train, col_indices, binary_categorical_columns
    )
    X_test = transform_columns(X_test, col_indices, binary_categorical_columns)

    stats = compute_statistics(
        X_train,
        col_indices,
        numerical_columns,
        categorical_columns,
        multi_class_categorical_columns,
    )

    X_train_array, columns = process_dataset(
        X_train,
        stats,
        numerical_columns,
        categorical_columns,
        multi_class_categorical_columns,
        col_indices,
    )
    X_test_array, _ = process_dataset(
        X_test,
        stats,
        numerical_columns,
        categorical_columns,
        multi_class_categorical_columns,
        col_indices,
    )

    return X_train_array, X_test_array, columns


def compute_statistics(
    X_train,
    col_indices,
    numerical_columns,
    categorical_columns,
    multi_class_categorical_columns,
):
    """
    Computes statistics (mean, std, mode) for numerical and categorical columns in X_train.

    Args:
        X_train (np.ndarray): training
        numerical_columns (list): numerical column names.
        categorical_columns (list): categorical column names.
        col_indices (dict): mapping of column names to indices in the data
        multi_class_categorical_columns (list): multi-class categorical columns.

    Returns:
        stats (dict): statistics for each column.
    """
    stats = {}
    for col in numerical_columns + categorical_columns:
        if col not in col_indices:
            raise ValueError(f"Column '{col}' not found in col_indices.")
        idx = col_indices[col]
        arr = X_train[:, idx]
        # Compute statistics, ignoring NaN values
        mode = compute_mode(arr)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        stats[col] = {
            "mode": mode,
            "mean": mean if mean else 0,
            "std": std if mean else 1,
        }
        if col in multi_class_categorical_columns:
            # Store unique values for one-hot encoding
            stats[col]["unique_values"] = np.unique(arr[~np.isnan(arr)])
    return stats


# Define paths
RAW_DATA_PATH = "data_raw"
CLEAN_DATA_PATH = "data_clean"


def main():
    """
    Main function to load data, clean it, and save the processed data.
    """
    parser = create_preprocessing_parser()

    args = parser.parse_args()
    if args.features == "fraction" and args.fraction_percentage is None:
        parser.error("--features fraction requires --fraction_percentage.")

    if args.features == "all":
        numerical_columns, categorical_columns = get_all_columns()
        binary_categorical_columns = get_all_binary_categorical_columns()
    elif args.features == "selected":
        numerical_columns, categorical_columns = get_selected_columns()
        binary_categorical_columns = get_selected_binary_categorical_columns()
    else:
        numerical_columns, categorical_columns, binary_categorical_columns = (
            get_random_column_subset(args.fraction_percentage)
        )

    x_train, x_test, y_train, train_ids, test_ids, col_indices = load_csv_data(
        RAW_DATA_PATH
    )

    print("Before cleaning")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)

    x_train, x_test, col_indices = clean_data(
        x_train,
        x_test,
        col_indices,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        binary_categorical_columns=binary_categorical_columns,
    )

    print("After cleaning")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)

    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)

    # Save processed data
    for data, name in zip(
        [x_train, x_test, y_train, train_ids, test_ids],
        ["x_train", "x_test", "y_train", "train_ids", "test_ids"],
    ):
        np.save(os.path.join(CLEAN_DATA_PATH, name + ".npy"), data)

    header = ", ".join([col for col in col_indices])

    with open(os.path.join(CLEAN_DATA_PATH, "header.txt"), "w") as f:
        f.write(header)


if __name__ == "__main__":
    main()
