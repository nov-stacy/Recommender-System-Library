import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from recommender_system_library.models.abstract._embeddings_recommender_system import EmbeddingDebug
from recommender_system_library.models.abstract import EmbeddingsRecommenderSystem


class TestEmbeddingDebug(unittest.TestCase):

    USER_INDICES, ITEM_INDICES, RATINGS = np.array([0, 1, 0]), np.array([0, 1, 2]), np.array([1, 1, 1])
    NOT_USER_INDICES_1, NOT_ITEM_INDICES_1, NOT_RATINGS_1 = np.array([[0, 1]]), np.array([[0, 1]]), np.array([[1, 1]])
    NOT_USER_INDICES_2, NOT_ITEM_INDICES_2, NOT_RATINGS_2 = np.array([0, 1]), np.array([0, 1]), np.array([1, 1])
    USER_MATRIX, ITEM_MATRIX = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1], [-1, 0]])
    NOT_USER_MATRIX_1, NOT_ITEM_MATRIX_1 = np.array([1, 0]), np.array([1, 0])
    NOT_USER_MATRIX_2, NOT_ITEM_MATRIX_2 = np.array([[1, 0, 0]]), np.array([[1, 0, 2], [-1, 0, 3]])
    MEAN_USERS, MEAN_ITEMS = np.array([[2 / 3], [1 / 3]]), np.array([[1 / 3], [1 / 3], [1 / 3]])
    NOT_MEAN_USERS_1, NOT_MEAN_ITEMS_1 = np.array([2 / 3]), np.array([1 / 3])
    NOT_MEAN_USERS_2, NOT_MEAN_ITEMS_2 = np.array([[2, 0], [1, 0]]), np.array([[1, 0], [1, 0], [1, 0]])
    NOT_MEAN_USERS_3, NOT_MEAN_ITEMS_3 = np.array([[2 / 3], [1 / 3], [0]]), np.array([[1 / 3], [1 / 3], [1 / 3], [0]])

    def test_create(self):
        embedding_debug = EmbeddingDebug()
        self.assertEqual(embedding_debug._debug_information, None)

    def test_update(self):
        embedding_debug = EmbeddingDebug()
        embedding_debug.update(True)
        self.assertEqual(embedding_debug._debug_information, list())
        embedding_debug.update(False)
        self.assertEqual(embedding_debug._debug_information, None)
        self.assertRaises(TypeError, embedding_debug.update, 1)

    def test_set(self):
        embedding_debug = EmbeddingDebug()
        embedding_debug.update(True)
        embedding_debug.set(self.USER_INDICES, self.ITEM_INDICES, self.RATINGS, self.USER_MATRIX,
                            self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertEqual(len(embedding_debug.get()), 1)
        self.assertLess(embedding_debug.get()[0], 2.5)
        values = [self.USER_INDICES, self.ITEM_INDICES, self.RATINGS, self.USER_MATRIX,
                  self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS]

        for index in range(len(values)):
            values_copy = values.copy()
            values_copy[index] = 1
            self.assertRaises(TypeError, embedding_debug.set, *values_copy)

        self.assertRaises(ValueError, embedding_debug.set, self.NOT_USER_INDICES_1, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.NOT_ITEM_INDICES_1, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.NOT_RATINGS_1,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.NOT_USER_INDICES_2, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.NOT_ITEM_INDICES_2, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.NOT_RATINGS_2,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.NOT_USER_MATRIX_1, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.NOT_ITEM_MATRIX_1, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.NOT_USER_MATRIX_2, self.ITEM_MATRIX, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.NOT_ITEM_MATRIX_2, self.MEAN_USERS, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.NOT_MEAN_USERS_1, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.NOT_MEAN_ITEMS_1)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.NOT_MEAN_USERS_2, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.NOT_MEAN_ITEMS_2)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.NOT_MEAN_USERS_3, self.MEAN_ITEMS)
        self.assertRaises(ValueError, embedding_debug.set, self.USER_INDICES, self.ITEM_INDICES, self.RATINGS,
                          self.USER_MATRIX, self.ITEM_MATRIX, self.MEAN_USERS, self.NOT_MEAN_ITEMS_3)

    def test_get(self):
        embedding_debug = EmbeddingDebug()
        self.assertRaises(AttributeError, embedding_debug.get)
        embedding_debug.update(True)
        self.assertEqual(embedding_debug.get(), list())


class TestEmbeddingsRecommenderSystem(unittest.TestCase):

    DATA = sparse.coo_matrix([[1, 2, 3], [1, 2, 3]])
    NEW_DATA = sparse.coo_matrix([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    DIMENSION = 2
    EPOCHS = 10
    USER_INDEX = 0
    ITEMS_COUNT = 1

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_create(self):
        for dimension in range(1, 100):
            embedding = EmbeddingsRecommenderSystem(dimension)
            self.assertEqual(dimension, embedding._dimension)
        self.assertRaises(TypeError, EmbeddingsRecommenderSystem, '1')
        self.assertRaises(ValueError, EmbeddingsRecommenderSystem, -1)

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_create_user_item_matrix(self):
        for dimension in range(1, 100):
            embedding = EmbeddingsRecommenderSystem(dimension)
            embedding._create_user_item_matrix(self.DATA)
            self.assertEqual(embedding._users_count, self.DATA.shape[0])
            self.assertEqual(embedding._items_count, self.DATA.shape[1])
            self.assertListEqual(list(embedding._user_matrix.shape), [self.DATA.shape[0], dimension])
            self.assertListEqual(list(embedding._item_matrix.shape), [self.DATA.shape[1], dimension])
            self.assertListEqual(list(embedding._mean_users), list(self.DATA.mean(axis=1)))
            self.assertListEqual(list(embedding._mean_items), list(self.DATA.mean(axis=0).transpose()))

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_create_information_for_debugging(self):
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        embedding._create_information_for_debugging(self.DATA, False)
        self.assertEqual(embedding._users_indices, None)
        self.assertEqual(embedding._items_indices, None)
        self.assertEqual(embedding._ratings, None)
        embedding._create_information_for_debugging(self.DATA, True)
        self.assertListEqual(list(embedding._users_indices), list(self.DATA.row))
        self.assertListEqual(list(embedding._items_indices), list(self.DATA.col))
        self.assertListEqual(list(embedding._ratings), list(self.DATA.data))

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_private_fit(self):
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        embedding._fit(self.EPOCHS, False)
        embedding.debug_information.update(True)
        embedding._create_user_item_matrix(self.DATA)
        embedding._create_information_for_debugging(self.DATA, True)
        embedding._fit(self.EPOCHS, True)

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_fit(self):
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, False)), EmbeddingsRecommenderSystem)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, True)), EmbeddingsRecommenderSystem)
        self.assertRaises(TypeError, embedding.fit, '1', self.EPOCHS)
        self.assertRaises(TypeError, embedding.fit, self.DATA, '1')
        self.assertRaises(TypeError, embedding.fit, self.DATA, self.EPOCHS, '1')
        self.assertRaises(ValueError, embedding.fit, self.DATA, -1)

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_refit(self):
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        self.assertRaises(AttributeError, embedding.refit, self.DATA, self.EPOCHS)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, False)), EmbeddingsRecommenderSystem)
        self.assertListEqual(list(embedding._user_matrix.shape), [self.DATA.shape[0], self.DIMENSION])
        self.assertListEqual(list(embedding._item_matrix.shape), [self.DATA.shape[1], self.DIMENSION])
        self.assertEqual(type(embedding.refit(self.NEW_DATA, self.EPOCHS, False)), EmbeddingsRecommenderSystem)
        self.assertListEqual(list(embedding._user_matrix.shape), [self.NEW_DATA.shape[0], self.DIMENSION])
        self.assertListEqual(list(embedding._item_matrix.shape), [self.NEW_DATA.shape[1], self.DIMENSION])
        self.assertRaises(TypeError, embedding.refit, '1', self.EPOCHS)
        self.assertRaises(TypeError, embedding.refit, self.DATA, '1')
        self.assertRaises(TypeError, embedding.refit, self.DATA, self.EPOCHS, '1')
        self.assertRaises(ValueError, embedding.refit, self.DATA, -1)

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_predict_ratings(self):
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        self.assertRaises(AttributeError, embedding.predict_ratings, self.USER_INDEX)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, False)), EmbeddingsRecommenderSystem)
        self.assertListEqual(list(embedding.predict_ratings(self.USER_INDEX).shape), [self.DATA.shape[1]])
        self.assertRaises(TypeError, embedding.predict_ratings, '1')
        self.assertRaises(ValueError, embedding.predict_ratings, -1)
        self.assertRaises(ValueError, embedding.predict_ratings, 100)

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_predict(self):
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        self.assertRaises(AttributeError, embedding.predict, self.USER_INDEX, self.ITEMS_COUNT)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, False)), EmbeddingsRecommenderSystem)
        self.assertListEqual(list(embedding.predict(self.USER_INDEX, self.ITEMS_COUNT).shape), [self.ITEMS_COUNT])
        self.assertRaises(TypeError, embedding.predict, '1', self.ITEMS_COUNT)
        self.assertRaises(ValueError, embedding.predict, -1, self.ITEMS_COUNT)
        self.assertRaises(ValueError, embedding.predict, 100, self.ITEMS_COUNT)
        self.assertRaises(TypeError, embedding.predict, self.USER_INDEX, '1')
        self.assertRaises(ValueError, embedding.predict, self.USER_INDEX, -1)
        self.assertRaises(ValueError, embedding.predict, self.USER_INDEX, 100)

    @mock.patch('recommender_system_library.models.abstract.EmbeddingsRecommenderSystem.__abstractmethods__', set())
    def test_str(self) -> None:
        embedding = EmbeddingsRecommenderSystem(self.DIMENSION)
        self.assertIn('embeddings', str(embedding))
