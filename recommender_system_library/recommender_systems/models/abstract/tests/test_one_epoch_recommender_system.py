import unittest
from unittest import mock

from scipy import sparse

from recommender_systems.models.abstract import TrainWithOneEpochARS


class TestAbstractRecommenderSystemTrainWithOneEpoch(unittest.TestCase):

    DATA = sparse.coo_matrix([[1, 2, 3], [1, 2, 3]])
    USER_INDEX = 0

    @mock.patch('recommender_systems.models.abstract.TrainWithOneEpochARS.__abstractmethods__', set())
    def test_fit(self):
        model = TrainWithOneEpochARS()
        model.fit(self.DATA)
        self.assertRaises(TypeError, model.fit, 1)

    @mock.patch('recommender_systems.models.abstract.TrainWithOneEpochARS.__abstractmethods__', set())
    def test_refit(self):
        model = TrainWithOneEpochARS()
        self.assertRaises(AttributeError, model.refit, self.DATA)
        self.assertEqual(type(model.fit(self.DATA)), TrainWithOneEpochARS)
        self.assertRaises(TypeError, model.refit, 1)

    @mock.patch('recommender_systems.models.abstract.TrainWithOneEpochARS.__abstractmethods__', set())
    def test_predict_ratings(self):
        model = TrainWithOneEpochARS()
        self.assertRaises(AttributeError, model.predict_ratings, self.USER_INDEX)
        self.assertEqual(type(model.fit(self.DATA)), TrainWithOneEpochARS)
        self.assertRaises(TypeError, model.predict_ratings, '1')
        self.assertRaises(ValueError, model.predict_ratings, -1)

    @mock.patch('recommender_systems.models.abstract.TrainWithOneEpochARS.__abstractmethods__', set())
    def test_predict(self):
        model = TrainWithOneEpochARS()
        self.assertRaises(AttributeError, model.predict, self.USER_INDEX)
        model.fit(self.DATA)
        self.assertRaises(TypeError, model.predict, '1')
        self.assertRaises(ValueError, model.predict, -1)

    @mock.patch('recommender_systems.models.abstract.TrainWithOneEpochARS.__abstractmethods__', set())
    def test_str(self):
        model = TrainWithOneEpochARS()
        self.assertIn('one epoch', str(model))
