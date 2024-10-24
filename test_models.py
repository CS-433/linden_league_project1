import unittest
import numpy as np
from models import SVM, KNN
import os
from helpers import load_csv_data, create_csv_submission_without_ids

class SVMTest(unittest.TestCase):
    def test_svm(self):
        x = np.array([[0, 0], [.1, .5], [.6, .7], [1, 1], [.2, .2]])
        y = np.array([0, 0, 1, 1, 0])
        svm = SVM()
        svm.fit(x, y)
        
        self.assertEquals(svm.predict(np.array([0,0])), 0)

class KNNTest(unittest.TestCase):
    def test_nearest(self):
        self.assertEqual(1, 1)
        x = np.array([
            [0.1, 0],
            [0.2, 0],
            [0.3, .2],
            [0.5, 0],
            [0.2, .2],
        ])
        y = np.array(range(len(x)))
        for k_minus, expected in enumerate([
            [0],
            [0, 1],
            [1, 0, 4],
            [0, 1, 2, 4],
        ]):

            model = KNN(k=k_minus+1)
            model.fit(x, y)
            np.testing.assert_equal(model.knn(np.array([0, 0])), expected)

    def test_nearest(self):
        self.assertEqual(1, 1)
        x = np.array([
            [0.1, 0],
            [0.2, 0],
            [0.3, .2],
            [0.5, 0],
            [0.2, .2],
        ])
        y = np.array(range(len(x)))
        for k_minus, expected in enumerate([
            [0],
            [0, 1],
            [1, 0, 4],
            [0, 1, 2, 4],
        ]):

            model = KNN(k=k_minus+1)
            model.fit(x, y)
            np.testing.assert_equal(model.knn(np.array([0, 0])), expected)

    def test_class(self):
        self.assertEqual(1, 1)
        x = np.array([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10],
            [11],
        ])
        y = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0])
        assert(len(x) == len(y))
        for k, expected in [(1, 0), (3, 0), (5, 1), (7, 1), (9, 1), (10, 0)]:
            model = KNN(k=k)
            model.fit(x, y)
            self.assertEqual(model.predict(np.array([0])), expected, "For k = %d" % k)

    def test_predict(self, k = 3):
        SAVE_DIR = "knn_pred"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        x_train, x_test, y_train = load_clean_data()
        one_percent = int(len(x_test) / 100) 

        model = KNN(k)
        model.fit(x_train, y_train)

        all_save_files = map(lambda x: list(map(int, x.split(".")[:-1])), filter(lambda x: x.endswith(".npy"), os.listdir(SAVE_DIR)))
        save_files = filter(lambda x: x[0] == k and x[2] == len(x_test),all_save_files)
        saved_is = list(map(lambda x: x[1],save_files))
        if len(saved_is) > 0:
            i = max(saved_is)
            y_pred = np.load(os.path.join(SAVE_DIR, f"{k}.{i}.{len(x_test)}.npy"))
            i+=1
        else:
            i = 0
            y_pred = np.zeros(x_test.shape[0])

        while i < len(x_test):
            y_pred[i] = model.predict(x_test[i])

            if i % one_percent == 0:
                print(f"{i/len(x_test)*100:.2f}%")
                filename = os.path.join(SAVE_DIR, f"{k}.{i}.{len(x_test)}.npy")
                np.save(filename, y_pred)
            i += 1

        filename = os.path.join(SAVE_DIR, f"{k}.{len(x_test)}.csv")
        create_csv_submission_without_ids(y_pred, filename)

    def test_one(self):
        x_train, _, y_train = load_clean_data()
        x_train_unique, unique_idx = np.unique(x_train, axis=0, return_index=True)
        y_train_unique = y_train[unique_idx]
        print("Total points: ", len(x_train))
        print("Unique points: ", len(x_train_unique))

        knn1 = KNN(k=1)
        knn1.fit(x_train_unique, y_train_unique)
        for _ in range(1000):
            i = np.random.randint(0, len(x_train_unique))
            assert knn1.predict(x_train_unique[i]) == y_train_unique[i]


if __name__ == "__main__":
    unittest.main()
