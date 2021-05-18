import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from recommender_systems.models.abstract import EmbeddingsARS


EmbeddingDebug = EmbeddingsARS._EmbeddingsARS__EmbeddingDebug


class TestEmbeddingDebug(unittest.TestCase):

    U_IND, I_IND, RATS = np.array([0, 1, 0]), np.array([0, 1, 2]), np.array([1, 1, 1])
    NOT_U_IND_1, NOT_I_IND_1, NOT_RATS_1 = np.array([[0, 1]]), np.array([[0, 1]]), np.array([[1, 1]])
    NOT_U_IND_2, NOT_I_IND_2, NOT_RATS_2 = np.array([0, 1]), np.array([0, 1]), np.array([1, 1])
    U_M, I_M = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1], [-1, 0]])
    NOT_U_M_1, NOT_I_M_1 = np.array([1, 0]), np.array([1, 0])
    NOT_U_M_2, NOT_I_M_2 = np.array([[1, 0, 0]]), np.array([[1, 0, 2], [-1, 0, 3]])

    def test_set(self):
        embedding_debug = EmbeddingDebug()
        embedding_debug._update('mse')
        embedding_debug._set(self.U_IND, self.I_IND, self.RATS, self.U_M, self.I_M)
        self.assertEqual(len(embedding_debug.get()), 1)
        self.assertLess(embedding_debug.get()[0], 2.5)
        values = [self.U_IND, self.I_IND, self.RATS, self.U_M, self.I_M]

        for index in range(len(values)):
            values_copy = values.copy()
            values_copy[index] = 1
            self.assertRaises(TypeError, embedding_debug._set, *values_copy)

        self.assertRaises(ValueError, embedding_debug._set, self.NOT_U_IND_1, self.I_IND, self.RATS, self.U_M, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.NOT_I_IND_1, self.RATS, self.U_M, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.I_IND, self.NOT_RATS_1, self.U_M, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.NOT_U_IND_2, self.I_IND, self.RATS, self.U_M, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.NOT_I_IND_2, self.RATS, self.U_M, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.I_IND, self.NOT_RATS_2, self.U_M, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.I_IND, self.RATS, self.NOT_U_M_1, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.I_IND, self.RATS, self.U_M, self.NOT_I_M_1)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.I_IND, self.RATS, self.NOT_U_M_2, self.I_M)
        self.assertRaises(ValueError, embedding_debug._set, self.U_IND, self.I_IND, self.RATS, self.U_M, self.NOT_I_M_2)

    def test_get(self):
        embedding_debug = EmbeddingDebug()
        self.assertRaises(AttributeError, embedding_debug.get)
        embedding_debug._update('mse')
        self.assertEqual(embedding_debug.get(), list())


class TestEmbeddingsARS(unittest.TestCase):

    DATA = sparse.coo_matrix([[1, 2, 3], [1, 2, 3]])
    NEW_DATA = sparse.coo_matrix([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    DIMENSION = 2
    EPOCHS = 10
    USER_INDEX = 0

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_create(self):
        for dimension in range(1, 100):
            embedding = EmbeddingsARS(dimension)
            self.assertEqual(dimension, embedding._dimension)
        self.assertRaises(TypeError, EmbeddingsARS, '1')
        self.assertRaises(ValueError, EmbeddingsARS, -1)

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_create_user_items_matrix(self):
        for dimension in range(1, 100):
            embedding = EmbeddingsARS(dimension)
            embedding._create_user_items_matrix(self.DATA)
            self.assertEqual(embedding._users_count, self.DATA.shape[0])
            self.assertEqual(embedding._items_count, self.DATA.shape[1])
            self.assertListEqual(list(embedding._users_matrix.shape), [self.DATA.shape[0], dimension])
            self.assertListEqual(list(embedding._items_matrix.shape), [self.DATA.shape[1], dimension])

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_create_information_for_debugging(self):
        embedding = EmbeddingsARS(self.DIMENSION)
        embedding._create_information_for_debugging(self.DATA, None)
        self.assertEqual(embedding._users_indices, None)
        self.assertEqual(embedding._items_indices, None)
        self.assertEqual(embedding._ratings, None)
        embedding._create_information_for_debugging(self.DATA, 'mse')
        self.assertListEqual(list(embedding._users_indices), list(self.DATA.row))
        self.assertListEqual(list(embedding._items_indices), list(self.DATA.col))
        self.assertListEqual(list(embedding._ratings), list(self.DATA.data))

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_fit(self):
        embedding = EmbeddingsARS(self.DIMENSION)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, None)), EmbeddingsARS)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, 'mse')), EmbeddingsARS)
        self.assertRaises(TypeError, embedding.fit, '1', self.EPOCHS)
        self.assertRaises(TypeError, embedding.fit, self.DATA, '1')
        self.assertRaises(ValueError, embedding.fit, self.DATA, self.EPOCHS, '1')
        self.assertRaises(ValueError, embedding.fit, self.DATA, -1)

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_refit(self):
        embedding = EmbeddingsARS(self.DIMENSION)
        self.assertRaises(AttributeError, embedding.refit, self.DATA, self.EPOCHS)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, None)), EmbeddingsARS)
        self.assertListEqual(list(embedding._users_matrix.shape), [self.DATA.shape[0], self.DIMENSION])
        self.assertListEqual(list(embedding._items_matrix.shape), [self.DATA.shape[1], self.DIMENSION])
        self.assertEqual(type(embedding.refit(self.NEW_DATA, self.EPOCHS, None)), EmbeddingsARS)
        self.assertListEqual(list(embedding._users_matrix.shape), [self.NEW_DATA.shape[0], self.DIMENSION])
        self.assertListEqual(list(embedding._items_matrix.shape), [self.NEW_DATA.shape[1], self.DIMENSION])
        self.assertRaises(TypeError, embedding.refit, '1', self.EPOCHS)
        self.assertRaises(TypeError, embedding.refit, self.DATA, '1')
        self.assertRaises(ValueError, embedding.refit, self.DATA, self.EPOCHS, '1')
        self.assertRaises(ValueError, embedding.refit, self.DATA, -1)

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_predict_ratings(self):
        embedding = EmbeddingsARS(self.DIMENSION)
        self.assertRaises(AttributeError, embedding.predict_ratings, self.USER_INDEX)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, None)), EmbeddingsARS)
        self.assertListEqual(list(embedding.predict_ratings(self.USER_INDEX).shape), [self.DATA.shape[1]])
        self.assertRaises(TypeError, embedding.predict_ratings, '1')
        self.assertRaises(ValueError, embedding.predict_ratings, -1)

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_predict(self):
        embedding = EmbeddingsARS(self.DIMENSION)
        self.assertRaises(AttributeError, embedding.predict, self.USER_INDEX)
        self.assertEqual(type(embedding.fit(self.DATA, self.EPOCHS, None)), EmbeddingsARS)
        self.assertListEqual(list(embedding.predict(self.USER_INDEX).shape), [self.DATA.shape[1]])
        self.assertRaises(TypeError, embedding.predict, '1')
        self.assertRaises(ValueError, embedding.predict, -1)

    @mock.patch('recommender_systems.models.abstract.EmbeddingsARS.__abstractmethods__', set())
    def test_str(self) -> None:
        embedding = EmbeddingsARS(self.DIMENSION)
        self.assertIn('embeddings', str(embedding))
