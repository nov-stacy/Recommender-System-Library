import unittest

import numpy as np
from scipy import sparse
import sklearn

from recommender_system_library.models.memory_based_models import UserBasedModel


class TestUserBasedModel(unittest.TestCase):

    DATA = sparse.coo_matrix([[1, 2, 3], [1, 2, 2], [0, 1, 3], [4, 5, 6]])
    NEW_DATA = sparse.coo_matrix([[1, 2, 3], [1, 2, 2], [0, 1, 3]])
    K_NEAREST_NEIGHBOURS = 1
    USER_INDEX = 0

    def test_create(self):
        for k_nearest_neighbours in range(1, 100):
            model = UserBasedModel(k_nearest_neighbours)
            self.assertTrue(np.array_equal(model._data.todense(), sparse.coo_matrix([]).todense()))
            self.assertListEqual(list(model._mean_users), list())
            self.assertListEqual(list(model._mean_items), list())
            self.assertEqual(k_nearest_neighbours, model._k_nearest_neighbours)
            self.assertEqual(model._knn.get_params()['n_neighbors'], k_nearest_neighbours + 1)
            self.assertRaises(sklearn.exceptions.NotFittedError, sklearn.utils.validation.check_is_fitted, model._knn)
        self.assertRaises(TypeError, UserBasedModel, '1')
        self.assertRaises(ValueError, UserBasedModel, -1)

    def test_fit(self):
        model = UserBasedModel(self.K_NEAREST_NEIGHBOURS)
        model.fit(self.DATA)
        self.assertTrue(np.array_equal(model._data.todense(), self.DATA.todense()))
        self.assertTrue(np.array_equal(model._mean_users, self.DATA.mean(axis=1)))
        self.assertTrue(np.array_equal(model._mean_items, self.DATA.mean(axis=0).transpose()))
        sklearn.utils.validation.check_is_fitted(model._knn)

    def test_refit(self):
        model = UserBasedModel(self.K_NEAREST_NEIGHBOURS)
        model.fit(self.DATA)
        model.refit(self.NEW_DATA)
        self.assertTrue(np.array_equal(model._data.todense(), self.NEW_DATA.todense()))
        self.assertTrue(np.array_equal(model._mean_users, self.NEW_DATA.mean(axis=1)))
        self.assertTrue(np.array_equal(model._mean_items, self.NEW_DATA.mean(axis=0).transpose()))
        sklearn.utils.validation.check_is_fitted(model._knn)

    def test_predict_ratings(self):
        model = UserBasedModel(self.K_NEAREST_NEIGHBOURS)
        model.fit(self.DATA)
        ratings = model.predict_ratings(self.USER_INDEX)
        self.assertTrue(type(ratings), np.ndarray)
        self.assertListEqual(list(ratings.shape), [self.DATA.shape[1]])

    def test_predict(self):
        model = UserBasedModel(self.K_NEAREST_NEIGHBOURS)
        model.fit(self.DATA)
        items = model.predict(self.USER_INDEX)
        self.assertTrue(type(items), np.ndarray)

    def test_str(self):
        model = UserBasedModel(self.K_NEAREST_NEIGHBOURS)
        self.assertIn('UBM', str(model))
