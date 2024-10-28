import os
import json
import numpy as np
import pickle
from copy import deepcopy

from columns import (
    get_all_binary_categorical_columns,
    get_all_columns,
    get_random_column_subset,
    get_selected_binary_categorical_columns,
    get_selected_columns,
)
from helpers import load_csv_data

from parser_helpers import create_preprocessing_parser
from models import PCA


### Define paths
RAW_DATA_PATH = "data_raw"
CLEAN_DATA_PATH = "data_clean"
FORMAT_PATH = "format.json"



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

    with open(FORMAT_PATH, "r") as f:
        data_format = json.load(f)

    zero_values = ["None", "Not at any time", "Never"]
    missing_values = [
        "Refused",
        "Don't know",
    ]
    for col, idx in col_indices.items():
        for value, description in data_format[col].items():
            arr = X[:, idx]

            description = description.strip()
            for zero_value in zero_values:
                if description.startswith(zero_value):
                    mask = np.isin(arr, [int(value)])
                    X[:, idx] = np.where(mask, 0, arr)

            for missing_value in missing_values:
                if missing_value in description:
                    mask = np.isin(arr, [int(value)])
                    X[:, idx] = np.where(mask, np.nan, arr)

    return X


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
    standardize_num=True,
    onehot_cat=True,
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
        standardize_num (bool): whether to standardize numerical columns
        onehot_cat (bool): whether to one-hot encode categorical columns

    Returns:
        X_array (np.ndarray): processed data
        new_col_indices (dict): mapping of column names to indices for processed data
    """
    X_new = []
    new_col_indices = {}
    col_idx = 0

    # Process numerical columns
    for col in numerical_columns:
        if np.isnan(stats[col]["mean"]):
            continue
        idx = col_indices[col]
        arr = X[:, idx]
        # Impute missing values with mean
        arr = np.where(np.isnan(arr), stats[col]["mean"], arr)

        # Standardize
        if standardize_num:
            arr = (arr - stats[col]["mean"]) / (stats[col]["std"] + 1e-7)

        X_new.append(arr.reshape(-1, 1))
        new_col_indices[col] = col_idx
        col_idx += 1

    # Process categorical columns
    for col in categorical_columns:
        if np.isnan(stats[col]["mode"]):
            continue
        idx = col_indices[col]
        arr = X[:, idx]
        # Impute missing values with mode
        arr = np.where(np.isnan(arr), stats[col]["mode"], arr)

        if col in multi_class_categorical_columns and onehot_cat:
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
    X_array = np.hstack(X_new) if len(X_new) > 0 else np.empty((X.shape[0], 0))

    return X_array, new_col_indices


def clean_data(
    X_train,
    X_test,
    col_indices,
    numerical_columns,
    categorical_columns,
    binary_categorical_columns,
    standardize_num=True,
    onehot_cat=True,
    skip_rule_transformations=False,
    eval_split_idx=None,
):
    """
    Cleans and processes the training and test datasets.

    Args:
        X_train (np.ndarray): training data
        X_test (np.ndarray): test data.
        col_indices (dict): mapping of column names to their indices in the data
        numerical_columns (list): numerical column names
        categorical_columns (list): categorical column names
        binary_categorical_columns (list): binary categorical column names
        standardize_num (bool): whether to standardize numerical columns
        onehot_cat (bool): whether to one-hot encode categorical columns
        skip_rule_transformations (bool): whether to skip rule-based transformations
        eval_split_idx (int): index to split the training data for evaluation (only affects statistics computation)

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

    if not skip_rule_transformations:
        X_train = transform_columns(
            X_train, col_indices, binary_categorical_columns
        )
        X_test = transform_columns(X_test, col_indices, binary_categorical_columns)

    stats = compute_statistics(
        X_train[:eval_split_idx] if eval_split_idx else X_train,
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
        standardize_num=standardize_num,
        onehot_cat=onehot_cat,
    )
    X_test_array, _ = process_dataset(
        X_test,
        stats,
        numerical_columns,
        categorical_columns,
        multi_class_categorical_columns,
        col_indices,
        standardize_num=standardize_num,
        onehot_cat=onehot_cat,
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


def load_clean_data(data_path=CLEAN_DATA_PATH):
    """
    Load cleaned data from the specified path.

    Args:
        data_path (str): path to the cleaned data.

    Returns:
        dict: dictionary containing the loaded data.
    """
    return {
        name: np.load(os.path.join(data_path, f"{name}.npy"))
        for name in ["x", "x_final", "y", "ids", "ids_final"]
    }


def get_pca_transformed_data(x, x_final, col_idx_mapping, cols_already_used, eval_split_idx=None, max_frac_of_nan=0.8, min_explained_variance=0.8):
    """
    Apply PCA on the remaining columns of the data.

    Args:
        x : np.ndarray(N, D) : data matrix
        x_final : np.ndarray(N, D) : final data matrix
        col_idx_mapping : dict : mapping of column names to indices
        cols_already_used : set : columns already used in the model
        eval_split_idx : int : index to split the training data for evaluation
        max_frac_of_nan : float : maximum fraction of nans in a column
        min_explained_variance : float : minimum explained variance by the PCA

    Returns:
        x_pca : np.ndarray(N, D) : transformed data matrix
        x_final_pca : np.ndarray(N, D) : transformed final data matrix
        cols_for_pca_map : dict : mapping of column names to indices for PCA transformed data
    """
    ### get subset of columns excluded
    cols_excluded = set(col_idx_mapping.keys()) - cols_already_used
    col_idxs_for_pca = []
    for i, (col, idx) in enumerate(col_idx_mapping.items()):
        ### also filter out columns with too many nans
        if col in cols_excluded and np.isnan(x[:, idx]).mean() < max_frac_of_nan:
            col_idxs_for_pca.append(idx)
    x_pca = x[:, col_idxs_for_pca]
    x_final_pca = x_final[:, col_idxs_for_pca]

    ### fill nans with mean from train
    for col_idx in range(x_pca.shape[1]):
        mean_train = np.nanmean(x_pca[:eval_split_idx, col_idx])
        x_pca[np.isnan(x_pca[:, col_idx]), col_idx] = mean_train
        x_final_pca[np.isnan(x_final_pca[:, col_idx]), col_idx] = mean_train

    ### transform all the data using PCA
    pca = PCA(n_components=None, min_explained_variance=min_explained_variance, standardize=True).fit(x_pca[:eval_split_idx])
    x_pca, x_final_pca = pca.transform(x_pca, use_prev_stats=True), pca.transform(x_final_pca, use_prev_stats=True)
    cols_for_pca_map = {f"pca_{i}": i for i in range(x_pca.shape[1])}

    return x_pca, x_final_pca, cols_for_pca_map


def get_all_data(cfg, process_cols="all", pca_kwargs=None, standardize_num=True, onehot_cat=True, skip_rule_transformations=False, verbose=True):
    """
    Load and clean data, apply PCA if specified, and save the processed data.

    Args:
        cfg : dict : configuration dictionary
        pca_kwargs : dict : kwargs for PCA
        standardize_num : bool : whether to standardize numerical columns
        onehot_cat : bool : whether to one-hot encode categorical columns
        skip_rule_transformations : bool : whether to skip rule-based transformations
        verbose : bool : verbosity

    Returns:
        x : np.ndarray(N, D) : training data
        x_final : np.ndarray(N, D) : final data
        y : np.ndarray(N) : labels
        ids : np.ndarray(N) : ids of training data
        ids_final : np.ndarray(N) : ids of final data
        col_idx_map : dict : mapping of column names to indices
        cleaned_col_idx_map : dict : mapping of column names to indices for cleaned data
    """
    if cfg["allow_load_clean_data"]:
        ### load already cleaned data
        try:
            if verbose: print("Loading clean data...")
            x, x_final, y, ids, ids_final = load_clean_data(data_path=cfg["clean_data_path"]).values()
            with open(os.path.join(cfg["clean_data_path"], "col_idx_map.pkl"), "rb") as f:
                col_idx_map = pickle.load(f)
            with open(os.path.join(cfg["clean_data_path"], "cleaned_col_idx_map.pkl"), "rb") as f:
                cleaned_col_idx_map = pickle.load(f)
            if verbose: print(f"  Final data: {x.shape=}, {x_final.shape=}")
            return x, x_final, y, ids, ids_final, col_idx_map, cleaned_col_idx_map
        except FileNotFoundError:
            if verbose: print("Clean data not found. Loading raw data and cleaning...")

    ### load raw data
    if verbose: print("Loading raw data...")
    npy_loaded = load_npy_data(cfg["raw_data_path"])
    if npy_loaded:
        ### load data from npy
        x, x_final, y, ids, ids_final, col_idx_map = npy_loaded
    else:
        ### load data from csv
        x, x_final, y, ids, ids_final, col_idx_map = load_csv_data(cfg["raw_data_path"])
    if verbose: print(f"  Raw data: {x.shape=}, {x_final.shape=}")

    ### select columns
    if process_cols == "all":
        numerical_columns, categorical_columns = get_all_columns()
        binary_categorical_columns = get_all_binary_categorical_columns()
    elif process_cols == "selected":
        numerical_columns, categorical_columns = get_selected_columns()
        binary_categorical_columns = get_selected_binary_categorical_columns()
    else:
        assert type(process_cols) in (float,int), "process_cols must be a percentage (int or float), or 'all', or 'selected'"
        numerical_columns, categorical_columns, binary_categorical_columns = (
            get_random_column_subset(process_cols)
        )

    ### shuffle train/eval data
    shuffle_idxs = np.random.default_rng(seed=cfg["seed"]).permutation(x.shape[0])
    x, y, ids = x[shuffle_idxs], y[shuffle_idxs], ids[shuffle_idxs]

    ### get data split idx for train/test
    eval_split_idx = int(x.shape[0] * (1 - cfg.get("eval_frac", 0)))

    ### clean data
    if verbose: print("Cleaning data...")
    cleaned_x, cleaned_x_final, cleaned_col_idx_map = clean_data(
        x,
        x_final,
        col_idx_map,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        binary_categorical_columns=binary_categorical_columns,
        standardize_num=standardize_num,
        onehot_cat=onehot_cat,
        skip_rule_transformations=skip_rule_transformations,
        eval_split_idx=eval_split_idx,
    )
    if verbose: print(f"  Clean data: {cleaned_x.shape=}, {cleaned_x_final.shape=}")

    ### apply PCA
    if pca_kwargs is not None:
        if verbose: print("  Applying PCA...")
        _pca_kwargs = deepcopy(pca_kwargs)
        cols_already_used = set(cleaned_col_idx_map.keys())
        if "all_cols" in _pca_kwargs.keys():
            cols_already_used = set()
            del _pca_kwargs["all_cols"]
        pca_x, pca_x_final, pca_col_idx_map = get_pca_transformed_data(x=x, x_final=x_final,
            col_idx_mapping=col_idx_map, cols_already_used=cols_already_used, eval_split_idx=eval_split_idx, **_pca_kwargs)
        if verbose: print(f"  PCA data: {pca_x.shape=}, {pca_x_final.shape=}")

        ### combine
        x = np.concatenate([cleaned_x, pca_x], axis=1)
        x_final = np.concatenate([cleaned_x_final, pca_x_final], axis=1)
        for k in pca_col_idx_map: # update the column index mapping
            pca_col_idx_map[k] += len(cleaned_col_idx_map)
        cleaned_col_idx_map.update(pca_col_idx_map)
    else:
        x, x_final = cleaned_x, cleaned_x_final
    
    if verbose: print(f"  Preprocessed data: {x.shape=}, {x_final.shape=}")

    ### remap labels to 0, 1 from -1, 1
    if cfg["remap_labels_to_01"]:
        y = (y + 1) // 2

    ### save processed data used for training
    if verbose: print("Saving clean data...")
    os.makedirs(cfg["clean_data_path"], exist_ok=True)
    for data, name in zip(
        [x, x_final, y, ids, ids_final],
        ['x', 'x_final', 'y', 'ids', 'ids_final']
    ):
        np.save(os.path.join(cfg["clean_data_path"], name + '.npy'), data)
    with open(os.path.join(cfg["clean_data_path"], "col_idx_map.pkl"), "wb") as f:
        pickle.dump(col_idx_map, f)
    with open(os.path.join(cfg["clean_data_path"], "cleaned_col_idx_map.pkl"), "wb") as f:
        pickle.dump(cleaned_col_idx_map, f)
    with open(os.path.join(cfg["clean_data_path"], "meta.json"), "w") as f:
        json.dump({
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "binary_categorical_columns": binary_categorical_columns,
            "eval_split_idx": eval_split_idx,
        }, f)

    return (
        (x[:eval_split_idx], x[eval_split_idx:]),
        (y[:eval_split_idx], y[eval_split_idx:]),
        (ids[:eval_split_idx], ids[eval_split_idx:]),
        col_idx_map, cleaned_col_idx_map,
        (x_final, ids_final)
    )


def resave_csv_as_npy(data_path, transform_values=True):
    """
    Resave the data in the specified path as numpy arrays.

    Args:
        data_path (str): path to the data.
    """
    x, x_final, y, ids, ids_final, col_indices = load_csv_data(data_path)

    ### transform specific values
    if transform_values:
        x = map_columns(x, col_indices)
        x_final = map_columns(x_final, col_indices)

    ### save
    for data, name in zip(
        [x, x_final, y, ids, ids_final],
        ['x', 'x_final', 'y', 'ids', 'ids_final']
    ):
        np.save(os.path.join(data_path, name + '.npy'), data)

    with open(os.path.join(data_path, "col_indices.pkl"), "wb") as f:
        pickle.dump(col_indices, f)


def load_npy_data(data_path):
    """
    Load data from the specified path.

    Args:
        data_path (str): path to the data.

    Returns:
        tuple: tuple containing the loaded data (x, x_final, y, ids, ids_final, col_indices).
    """
    try:
        with open(os.path.join(data_path, "col_indices.pkl"), "rb") as f:
            col_indices = pickle.load(f)
        data = [np.load(os.path.join(data_path, f"{name}.npy")) for name in ["x", "x_final", "y", "ids", "ids_final"]]
        return *data, col_indices
    except FileNotFoundError:
        return None
