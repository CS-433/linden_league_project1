import unittest

from columns import (
    get_selected_binary_categorical_columns,
    get_selected_columns,
)
from data_preprocessing import clean_data
from helpers import load_csv_data

RAW_DATA_PATH = "data_raw"


class TestImplementations(unittest.TestCase):
    def test_data_preprocessing(self):
        x_train, x_test, y_train, train_ids, test_ids, col_indices = (
            load_csv_data(RAW_DATA_PATH)
        )
        numerical_columns, categorical_columns = get_selected_columns()
        binary_categorical_columns = get_selected_binary_categorical_columns()

        assert x_train.shape == (328135, 321)
        assert x_test.shape == (109379, 321)

        x_train, x_test, col_indices = clean_data(
            x_train,
            x_test,
            col_indices,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            binary_categorical_columns=binary_categorical_columns,
        )
        print("Training data shape: ", x_train.shape)
        assert x_train.shape == (328135, 85)

        print("Test data shape: ", x_train.shape)
        assert x_test.shape == (109379, 85)


if __name__ == "__main__":
    unittest.main()
