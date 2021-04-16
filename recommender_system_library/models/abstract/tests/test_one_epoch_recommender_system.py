import unittest
from unittest import mock

from scipy import sparse

from recommender_system_library.models.abstract import AbstractRecommenderSystemTrainWithOneEpoch


class TestAbstractRecommenderSystemTrainWithOneEpoch(unittest.TestCase):

    DATA = sparse.coo_matrix([[1, 2, 3], [1, 2, 3]])
    USER_INDEX = 0
    ITEMS_COUNT = 1

    @mock.patch('recommender_system_library.models.abstract.AbstractRecommenderSystemTrainWithOneEpoch'
                '.__abstractmethods__', set())
    def test_fit(self):
        model = AbstractRecommenderSystemTrainWithOneEpoch()
        model.fit(self.DATA)
        self.assertRaises(TypeError, model.fit, 1)

    @mock.patch('recommender_system_library.models.abstract.AbstractRecommenderSystemTrainWithOneEpoch'
                '.__abstractmethods__', set())
    def test_refit(self):
        model = AbstractRecommenderSystemTrainWithOneEpoch()
        self.assertRaises(AttributeError, model.refit, self.DATA)
        self.assertEqual(type(model.fit(self.DATA)), AbstractRecommenderSystemTrainWithOneEpoch)
        self.assertRaises(TypeError, model.refit, 1)

    @mock.patch('recommender_system_library.models.abstract.AbstractRecommenderSystemTrainWithOneEpoch'
                '.__abstractmethods__', set())
    def test_predict_ratings(self):
        model = AbstractRecommenderSystemTrainWithOneEpoch()
        self.assertRaises(AttributeError, model.predict_ratings, self.USER_INDEX)
        self.assertEqual(type(model.fit(self.DATA)), AbstractRecommenderSystemTrainWithOneEpoch)
        self.assertRaises(TypeError, model.predict_ratings, '1')
        self.assertRaises(ValueError, model.predict_ratings, -1)

    @mock.patch('recommender_system_library.models.abstract.AbstractRecommenderSystemTrainWithOneEpoch'
                '.__abstractmethods__', set())
    def test_predict(self):
        model = AbstractRecommenderSystemTrainWithOneEpoch()
        self.assertRaises(AttributeError, model.predict, self.USER_INDEX, self.ITEMS_COUNT)
        model.fit(self.DATA)
        self.assertRaises(TypeError, model.predict, '1', self.ITEMS_COUNT)
        self.assertRaises(ValueError, model.predict, -1, self.ITEMS_COUNT)
        self.assertRaises(TypeError, model.predict, self.USER_INDEX, '1')
        self.assertRaises(ValueError, model.predict, self.USER_INDEX, -1)

    @mock.patch('recommender_system_library.models.abstract.AbstractRecommenderSystemTrainWithOneEpoch'
                '.__abstractmethods__', set())
    def test_str(self):
        model = AbstractRecommenderSystemTrainWithOneEpoch()
        self.assertIn('one epoch', str(model))
