import os
import numpy as np
import pickle

from columns import get_binary_categorical_columns, get_columns
from helpers import load_csv_data
from models import PCA


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

def get_pca_transformed_data(x, x_final, col_idx_mapping, cols_already_used, max_frac_of_nan=0.8, min_explained_variance=0.7):
    """
    Apply PCA on the remaining columns of the data.

    Args:
        x : np.ndarray(N, D) : data matrix
        x_final : np.ndarray(N, D) : final data matrix
        col_idx_mapping : dict : mapping of column names to indices
        cols_already_used : set : columns already used in the model
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
        mean_train = np.nanmean(x_pca[:, col_idx])
        x_pca[np.isnan(x_pca[:, col_idx]), col_idx] = mean_train
        x_final_pca[np.isnan(x_final_pca[:, col_idx]), col_idx] = mean_train

    ### transform all the data using PCA
    pca = PCA(n_components=None, min_explained_variance=min_explained_variance).fit(x_pca)
    x_pca, x_final_pca = pca.transform(x_pca), pca.transform(x_final_pca)
    cols_for_pca_map = {f"pca_{i}": i for i in range(x_pca.shape[1])}

    return x_pca, x_final_pca, cols_for_pca_map

def get_all_data(cfg, pca_kwargs=None, verbose=True):
    """
    Load and clean data, apply PCA if specified, and save the processed data.

    Args:
        cfg : dict : configuration dictionary
        pca_kwargs : dict : kwargs for PCA
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

    ### load and clean data
    if verbose: print("Loading raw data...")
    npy_loaded = load_npy_data(cfg["raw_data_path"])
    if npy_loaded:
        ### load data from npy
        x, x_final, y, ids, ids_final, col_idx_map = npy_loaded
    else:
        ### load data from csv
        x, x_final, y, ids, ids_final, col_idx_map = load_csv_data(cfg["raw_data_path"])
    if verbose: print(f"  Raw data: {x.shape=}, {x_final.shape=}")
    if verbose: print("Cleaning data...")
    cleaned_x, cleaned_x_final, cleaned_col_idx_map = clean_data(x, x_final, col_idx_map)
    if verbose: print(f"  Clean data: {cleaned_x.shape=}, {cleaned_x_final.shape=}")

    ### apply PCA on the remaining columns
    if pca_kwargs is not None:
        if verbose: print("  Applying PCA on the remaining columns...")
        pca_x, pca_x_final, pca_col_idx_map = get_pca_transformed_data(x=x, x_final=x_final,
            col_idx_mapping=col_idx_map, cols_already_used=cleaned_col_idx_map.keys(), **pca_kwargs)
        if verbose: print(f"  PCA data: {pca_x.shape=}, {pca_x_final.shape=}")
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

    return x, x_final, y, ids, ids_final, col_idx_map, cleaned_col_idx_map

def resave_csv_as_npy(data_path):
    """
    Resave the data in the specified path as numpy arrays.

    Args:
        data_path (str): path to the data.
    """
    x, x_final, y, ids, ids_final, col_indices = load_csv_data(data_path)
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
