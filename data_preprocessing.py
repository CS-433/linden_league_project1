import os
import numpy as np

from columns import get_binary_categorical_columns, get_columns
from helpers import load_csv_data


### Define paths
RAW_DATA_PATH = "data_raw"
CLEAN_DATA_PATH = "data_clean"


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

def transform_special_cases(X, col_indices):
    """
    Applies special transformations to specific columns in X.

    Args:
        X (np.ndarray): data
        col_indices (dict): mapping of column names to indices for data
    """
    # Transform "_AGEG5YR"
    col = "_AGEG5YR"
    idx = col_indices[col]
    arr = X[:, idx]
    X[:, idx] = np.where(
        (arr > 0) & (arr < 14),
        arr,
        np.where(arr == 14, 0, np.nan)
    )

    # Transform "PHYSHLTH" and "MENTHLTH"
    for col in ["PHYSHLTH", "MENTHLTH"]:
        idx = col_indices[col]
        arr = X[:, idx]
        X[:, idx] = np.where(
            (arr >= 1) & (arr <= 30),
            arr,
            np.where(arr == 88, 0, np.nan)
        )

    # Transform "CHILDREN"
    col = "CHILDREN"
    idx = col_indices[col]
    arr = X[:, idx]
    X[:, idx] = np.where(
        (arr > 0) & (arr < 88),
        arr,
        np.where(arr == 88, 0, np.nan)
    )

    # Transform "_DRNKWEK"
    col = "_DRNKWEK"
    idx = col_indices[col]
    arr = X[:, idx]
    X[:, idx] = np.where(arr < 99900, arr, np.nan)

    # Transform fruit and vegetable consumption columns
    for col in ["FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_"]:
        idx = col_indices[col]
        arr = X[:, idx]
        X[:, idx] = np.where(arr < 9999, arr, np.nan)

def transform_multi_class_columns(X, col_indices, columns, valid_values, adjustment=0):
    """
    Transforms multi-class categorical columns in X.

    Args:
        X (np.ndarray): data
        col_indices (dict): mapping of column names to indices for data
        columns (list): column names
        valid_values (iterable): valid values to keep
        adjustment (int): value to subtract from valid entries
    """
    for col in columns:
        idx = col_indices[col]
        arr = X[:, idx]
        mask = np.isin(arr, valid_values)
        X[:, idx] = np.where(mask, arr - adjustment, np.nan)

def transform_columns(X, col_indices):
    """
    Transforms columns in the dataset X according to specified rules.
    Rules are based on answers in the questionnaire used for generation of the dataset.
    See https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf to better understand the transformations.

    Args:
        X (np.ndarray): data
        col_indices (dict): mapping of column names to indices for data

    Returns:
        X (np.ndarray): transformed data array
    """
    # Copy X to avoid modifying original data
    X = X.copy()

    binary_categorical_columns = get_binary_categorical_columns()

    transform_binary_columns(X, col_indices, binary_categorical_columns)

    transform_special_cases(X, col_indices)

    transform_multi_class_columns(X, col_indices, ["GENHLTH"], range(1, 6), adjustment=1)
    transform_multi_class_columns(X, col_indices, ["_RACE"], range(1, 9), adjustment=1)
    transform_multi_class_columns(X, col_indices, ["_SMOKER3"], range(1, 5), adjustment=1)
    transform_multi_class_columns(X, col_indices, ["EMPLOY1"], range(1, 9), adjustment=1)
    transform_multi_class_columns(X, col_indices, ["CHOLCHK"], range(1, 5), adjustment=1)
    transform_multi_class_columns(X, col_indices, ["EDUCA"], range(1, 7), adjustment=1)
    transform_multi_class_columns(X, col_indices, ["INCOME2"], range(1, 9), adjustment=1)

    return X

def process_dataset(
    X, stats, numerical_columns, categorical_columns,
    multi_class_categorical_columns, col_indices
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
            unique_values = stats[col]['unique_values']
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

def clean_data(X_train, X_test, col_indices):
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
    numerical_columns, categorical_columns = get_columns(split=True)
    binary_categorical_columns = get_binary_categorical_columns()
    multi_class_categorical_columns = [
        col for col in categorical_columns if col not in binary_categorical_columns
    ]
    stats = {}

    X_train = transform_columns(X_train, col_indices)
    X_test = transform_columns(X_test, col_indices)

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
        col_indices
    )
    X_test_array, _ = process_dataset(
        X_test, 
        stats, 
        numerical_columns, 
        categorical_columns,
        multi_class_categorical_columns, 
        col_indices
    )

    return X_train_array, X_test_array, columns

def compute_statistics(X_train, col_indices, numerical_columns, categorical_columns, multi_class_categorical_columns):
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
        stats[col] = {'mode': mode, 'mean': mean, 'std': std}
        if col in multi_class_categorical_columns:
            # Store unique values for one-hot encoding
            stats[col]['unique_values'] = np.unique(arr[~np.isnan(arr)])
    return stats

def load_clean_data(data_path=CLEAN_DATA_PATH):
    return {
        name: np.load(os.path.join(data_path, f"{name}.npy"))
        for name in ["x", "x_final", "y", "ids", "ids_final"]
    }

def main():
    """
    Main function to load data, clean it, and save the processed data.
    """
    x_train, x_test, y_train, train_ids, test_ids, col_indices = load_csv_data(
        RAW_DATA_PATH
    )

    print("Before cleaning")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)

    x_train, x_test, col_indices = clean_data(x_train, x_test, col_indices)

    print("After cleaning")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)

    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)

    # Save processed data
    for data, name in zip(
        [x_train, x_test, y_train, train_ids, test_ids],
        ['x_train', 'x_test', 'y_train', 'train_ids', 'test_ids']
    ):
        np.save(os.path.join(CLEAN_DATA_PATH, name + '.npy'), data)

    header = ', '.join([col for col in col_indices])

    with open(os.path.join(CLEAN_DATA_PATH, "header.txt"), "w") as f:
        f.write(header)

if __name__ == "__main__":
    main()
