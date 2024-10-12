import os

from data_preprocessing import clean_data
from helpers import load_csv_data


def main():
    data_path = os.path.join("data")
    x_train, x_test, y_train, train_ids, test_ids, col_indices = load_csv_data(
        data_path
    )
    print(x_train.shape)
    print(x_test.shape)
    x_train, x_test = clean_data(x_train, x_test, col_indices)
    print(x_train.shape)
    print(x_test.shape)


if __name__ == "__main__":
    main()
