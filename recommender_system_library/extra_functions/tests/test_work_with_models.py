import os
import pickle
import unittest

from recommender_system_library.extra_functions.work_with_models import *
from recommender_system_library.models.memory_based_models import UserBasedModel


class TestCreateModel(unittest.TestCase):

    MEMORY_BASED_PARAMS = {'k_nearest_neighbours': 10}
    SGD_PARAMS = {'dimension': 10, 'learning_rate': 0.1, 'user_regularization': 0.2, 'item_regularization': 0.3}
    DIMENSION_PARAMS = {'dimension': 10}
    NOT_MODEL = 'model'
    NOT_PARAMS = {'parameter': 'value'}

    def test_correct_behavior(self):
        for model_name, model_class in zip(MODELS_NAMES, MODELS_CLASSES):
            parameters = self.MEMORY_BASED_PARAMS if 'based' in model_name else self.SGD_PARAMS if 'sgd' in model_name \
                         else self.DIMENSION_PARAMS
            model = create_model(model_name, parameters)
            self.assertIsInstance(model, model_class)

    def test_raise_type_error(self):
        for model_name in MODELS_NAMES:
            self.assertRaises(TypeError, create_model, model_name, self.NOT_PARAMS)

    def test_raise_value_error(self):
        self.assertRaises(ValueError, create_model, self.NOT_MODEL, self.NOT_PARAMS)


class TestSaveModelToFile(unittest.TestCase):

    MODEL = UserBasedModel(k_nearest_neighbours=10)
    NOT_MODEL = [[1, 2], [3, 4]]
    PATH = 'model.pickle'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self):
        save_model_to_file(self.MODEL, self.PATH)

    def test_raise_type_error(self):
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

    def test_correct_behavior(self):
        with open(self.PATH, 'wb') as file:
            pickle.dump(self.MODEL, file)
        model = get_model_from_file(self.PATH)
        self.assertIsInstance(model, self.MODEL.__class__)
        self.assertEqual(str(model), str(self.MODEL))

    def test_raise_type_error(self):
        self.assertRaises(TypeError, get_model_from_file, self.NOT_PATH_2)

    def test_raise_value_error(self):
        with open(self.NOT_PATH_1, 'wb') as file:
            pickle.dump(list(), file)
        self.assertRaises(ValueError, get_model_from_file, self.NOT_PATH_1)
