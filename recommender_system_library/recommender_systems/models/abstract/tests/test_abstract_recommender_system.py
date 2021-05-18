import unittest
import unittest.mock as mock

from recommender_systems.models.abstract import AbstractRecommenderSystem


class TestAbstractRecommenderSystem(unittest.TestCase):

    @mock.patch('recommender_systems.models.abstract.AbstractRecommenderSystem.__abstractmethods__', set())
    def test_is_trained(self) -> None:
        model = AbstractRecommenderSystem()
        self.assertFalse(model._is_trained)
        self.assertFalse(model.is_trained)

    @mock.patch('recommender_systems.models.abstract.AbstractRecommenderSystem.__abstractmethods__', set())
    def test_is_predict(self) -> None:
        model = AbstractRecommenderSystem()
        self.assertRaises(AttributeError, model._check_trained_and_rise_error)

    @mock.patch('recommender_systems.models.abstract.AbstractRecommenderSystem.__abstractmethods__', set())
    def test_str(self) -> None:
        model = AbstractRecommenderSystem()
        self.assertIn('Abstract', str(model))
