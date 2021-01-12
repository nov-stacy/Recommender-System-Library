import unittest
import numpy as np
from recommender_system.simple_methods import *

DATA = np.array([[1, np.nan, np.nan], [1, 1, 2], [3, 2, 1], [0, 0, 0], [2, 2, 2]])


class NearestNeigborsMethodTestCase(unittest.TestCase):
    def test_create(self):
        self.assertIsInstance(NearestNeigborsMethod(2), NearestNeigborsMethod)

    def test_train_is_instance(self):
        self.assertIsInstance(NearestNeigborsMethod(2), NearestNeigborsMethod)

    def test_issue_ranked_list_k_items(self):
        for index in [1, 2]:
            model = NearestNeigborsMethod(1).train(DATA)
            self.assertEqual(model.issue_ranked_list(0, index).shape[0], index)

    def test_issue_ranked_list(self):
        self.assertEqual(list(NearestNeigborsMethod(1).train(DATA).issue_ranked_list(0, 2)), [1, 2])


if __name__ == '__main__':
    unittest.main()
