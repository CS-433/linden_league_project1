import unittest

from data_preprocessing import clean_data
from helpers import load_csv_data

RAW_DATA_PATH = "data_raw"


class TestImplementations(unittest.TestCase):
    def test_data_preprocessing(self):
        x_train, x_test, y_train, train_ids, test_ids, col_indices = (
            load_csv_data(RAW_DATA_PATH)
        )

        assert x_train.shape == (328135, 321)
        assert x_test.shape == (109379, 321)

        x_train, x_test, col_indices = clean_data(x_train, x_test, col_indices)
        assert x_train.shape == (328135, 86)
        assert x_test.shape == (109379, 86)


if __name__ == "__main__":
    unittest.main()
