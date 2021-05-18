import os
import pickle
import unittest

from recommender_systems.extra_functions.work_with_models import *
from recommender_systems.models.memory_based_models import UserBasedModel


class TestSaveModelToFile(unittest.TestCase):

    MODEL = UserBasedModel(k_nearest_neighbours=10)
    NOT_MODEL = [[1, 2], [3, 4]]
    PATH = 'model.pickle'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self) -> None:
        save_model_to_file(self.MODEL, self.PATH)

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, save_model_to_file, self.NOT_MODEL, self.PATH)
        self.assertRaises(TypeError, save_model_to_file, self.MODEL, self.NOT_PATH)


class TestGetModelFromFile(unittest.TestCase):

    MODEL = UserBasedModel(k_nearest_neighbours=10)
    PATH = 'model.pickle'
    NOT_PATH_1 = 'not_model.pickle'
    NOT_PATH_2 = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)
        if os.path.exists(self.NOT_PATH_1):
            os.remove(self.NOT_PATH_1)

    def test_correct_behavior(self) -> None:
        with open(self.PATH, 'wb') as file:
            pickle.dump(self.MODEL, file)
        model = get_model_from_file(self.PATH)
        self.assertIsInstance(model, self.MODEL.__class__)
        self.assertEqual(str(model), str(self.MODEL))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, get_model_from_file, self.NOT_PATH_2)

    def test_raise_value_error(self) -> None:
        with open(self.NOT_PATH_1, 'wb') as file:
            pickle.dump(list(), file)
        self.assertRaises(ValueError, get_model_from_file, self.NOT_PATH_1)
