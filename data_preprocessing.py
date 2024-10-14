import os
import numpy as np

from helpers import load_csv_data


def get_columns(split=False):
    categorical_columns = [
        "HLTHPLN1",
        "MEDCOST",
        "EDUCA",
        "INCOME2",
        "CHOLCHK",
        "PNEUVAC3",
        "EMPLOY1",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        "USEEQUIP",
        "_AGEG5YR",
        "GENHLTH",
        "_TOTINDA",
        "_RACE",
        "CVDSTRK3",
        "ASTHMA3",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "_SMOKER3",
        "_RFSMOK3",
        "DRNKANY5",
        "_RFBING5",
        "_RFDRHV5",
        "_RFHYPE5",
        "_RFCHOL",
        "DIABETE3",
    ]

    numerical_columns = [
        "_BMI5",
        "CHILDREN",
        "FTJUDA1_",
        "FRUTDA1_",
        "BEANDAY_",
        "GRENDAY_",
        "ORNGDAY_",
        "VEGEDA1_",
        "PHYSHLTH",
        "MENTHLTH",
        "HTM4",
        "WTKG3",
        "_DRNKWEK",
    ]
    if split:
        return numerical_columns, categorical_columns
    return numerical_columns + categorical_columns

def get_binary_categorical_columns():
    binary_categorical_columns = [
        "HLTHPLN1",
        "MEDCOST",
        "PNEUVAC3",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        "USEEQUIP",
        "_TOTINDA",
        "_RACE",
        "CVDSTRK3",
        "ASTHMA3",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "_RFSMOK3",
        "DRNKANY5",
        "_RFBING5",
        "_RFDRHV5",
        "_RFHYPE5",
        "_RFCHOL",
        "DIABETE3",
    ]
    return binary_categorical_columns

def compute_mode(arr):
    values, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
    if len(counts) == 0:
        return np.nan
    max_count_index = np.argmax(counts)
    return values[max_count_index]

def transform_columns(X, col_indices):
    # Copy X to avoid modifying original data
    X = X.copy()

    binary_categorical_columns = get_binary_categorical_columns()

    def transform_binary_columns(columns):
        for col in columns:
            idx = col_indices[col]
            arr = X[:, idx]
            mask = np.isin(arr, [1, 2])
            X[:, idx] = np.where(mask, arr - 1, np.nan)

    def transform_special_cases():
        col = "_AGEG5YR"
        idx = col_indices[col]
        arr = X[:, idx]
        X[:, idx] = np.where((arr > 0) & (arr < 14), arr, np.where(arr == 14, 0, np.nan))

        for col in ["PHYSHLTH", "MENTHLTH"]:
            idx = col_indices[col]
            arr = X[:, idx]
            X[:, idx] = np.where((arr >= 1) & (arr <= 30), arr, np.where(arr == 88, 0, np.nan))

        col = "CHILDREN"
        idx = col_indices[col]
        arr = X[:, idx]
        X[:, idx] = np.where((arr > 0) & (arr < 88), arr, np.where(arr == 88, 0, np.nan))

        col = "_DRNKWEK"
        idx = col_indices[col]
        arr = X[:, idx]
        X[:, idx] = np.where(arr < 99900, arr, np.nan)

        for col in ["FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_"]:
            idx = col_indices[col]
            arr = X[:, idx]
            X[:, idx] = np.where(arr < 9999, arr, np.nan)

    def transform_multi_class_columns(columns, valid_values, adjustment=0):
        for col in columns:
            idx = col_indices[col]
            arr = X[:, idx]
            mask = np.isin(arr, valid_values)
            X[:, idx] = np.where(mask, arr - adjustment, np.nan)

    transform_binary_columns(binary_categorical_columns)
    transform_special_cases()
    transform_multi_class_columns(["GENHLTH"], range(1, 6), adjustment=1)
    transform_multi_class_columns(["_RACE"], range(1, 9), adjustment=1)
    transform_multi_class_columns(["_SMOKER3"], range(1, 5), adjustment=1)
    transform_multi_class_columns(["EMPLOY1"], range(1, 9), adjustment=1)
    transform_multi_class_columns(["CHOLCHK"], range(1, 5), adjustment=1)
    transform_multi_class_columns(["EDUCA"], range(1, 7), adjustment=1)
    transform_multi_class_columns(["INCOME2"], range(1, 9), adjustment=1)

    return X

def clean_data(X_train, X_test, col_indices):
    numerical_columns, categorical_columns = get_columns(split=True)
    binary_categorical_columns = get_binary_categorical_columns()
    multi_class_categorical_columns = [col for col in categorical_columns if col not in binary_categorical_columns]
    stats = {}

    def process_dataset(X, stats):
        X_new = []
        new_col_names = []

        for col in numerical_columns:
            idx = col_indices[col]
            arr = X[:, idx]
            arr = np.where(np.isnan(arr), stats[col]["mean"], arr)
            arr = (arr - stats[col]["mean"]) / stats[col]["std"]
            X_new.append(arr.reshape(-1, 1))
            new_col_names.append(col)

        for col in categorical_columns:
            idx = col_indices[col]
            arr = X[:, idx]
            arr = np.where(np.isnan(arr), stats[col]["mode"], arr)

            if col in multi_class_categorical_columns:
                unique_values = stats[col]['unique_values']
                for val in unique_values:
                    new_col_name = f"{col}_{val}"
                    new_col_names.append(new_col_name)
                    one_hot_arr = (arr == val).astype(float).reshape(-1, 1)
                    X_new.append(one_hot_arr)
            else:
                X_new.append(arr.reshape(-1, 1))
                new_col_names.append(col)

        # Stack the columns to form a 2D array
        X_array = np.hstack(X_new)

        return X_array, new_col_names

    X_train = transform_columns(X_train, col_indices)
    X_test = transform_columns(X_test, col_indices)

    for col in numerical_columns + categorical_columns:
        idx = col_indices[col]
        arr = X_train[:, idx]
        mode = compute_mode(arr)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        stats[col] = {'mode': mode, 'mean': mean, 'std': std}
        if col in multi_class_categorical_columns:
            stats[col]['unique_values'] = np.unique(arr[~np.isnan(arr)]) 

    X_train_array, columns = process_dataset(X_train, stats)
    X_test_array, _ = process_dataset(X_test, stats)

    return X_train_array, X_test_array


RAW_DATA_PATH = "data_raw"
CLEAN_DATA_PATH = "data_clean"
def main():
    x_train, x_test, y_train, train_ids, test_ids, col_indices = load_csv_data(
        RAW_DATA_PATH
    )

    print("Before cleaning")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)
    x_train, x_test = clean_data(x_train, x_test, col_indices)
    print("After cleaning")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)

    for data, name in zip([x_train, x_test, y_train, train_ids, test_ids], ['x_train', 'x_test', 'y_train', 'train_ids', 'test_ids']):
        path = os.path.join(CLEAN_DATA_PATH, name)
        np.save(os.path.join(CLEAN_DATA_PATH, name + '.npy'), data)


if __name__ == "__main__":
    main()

