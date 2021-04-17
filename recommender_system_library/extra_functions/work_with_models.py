import typing as tp

import pickle

from utilspie.collectionsutils import frozendict

from recommender_system_library.models.abstract import AbstractRecommenderSystem
from recommender_system_library.models import *


__all__ = [
    'MODELS_NAMES',
    'MODELS_CLASSES',
    'create_model',
    'save_model_to_file',
    'get_model_from_file'
]


MODELS_NAMES = tuple([
    'user_based_model', 'item_based_model', 'latent_factor_als_model', 'latent_factor_hals_model',
    'latent_factor_sgd_model', 'latent_factor_svd_model', 'implicit_sgd_model'
])

MODELS_CLASSES = tuple([
    memory_based_models.UserBasedModel, memory_based_models.ItemBasedModel,
    latent_factor_models.AlternatingLeastSquaresModel, latent_factor_models.HierarchicalAlternatingLeastSquaresModel,
    latent_factor_models.StochasticLatentFactorModel, latent_factor_models.SingularValueDecompositionModel,
    implicit_models.ImplicitStochasticLatentFactorModel
])


MODELS_DICT: tp.Dict[str, AbstractRecommenderSystem.__class__] = frozendict(zip(MODELS_NAMES, MODELS_CLASSES))


def create_model(type_model: str, parameters: tp.Dict[str, tp.Any]) -> AbstractRecommenderSystem:
    """
    Method to create model from name and dictionary of parameters

    Parameters
    ----------
    type_model: str
        Name of model (user_based_model, item_based_model, latent_factor_als_model, latent_factor_hals_model,
                       latent_factor_sgd_model, latent_factor_svd_model, implicit_sgd_model)
    parameters: dictionary str -> object
        Parameters for creating model

    Raises
    ------
    KeyError
        If type model is not supported
    TypeError
        If parameters do not fit the model

    Returns
    -------
    model: AbstractRecommenderSystem
    """

    if type_model not in MODELS_NAMES:
        raise ValueError('There is no such model')

    return MODELS_DICT[type_model](**parameters)


def save_model_to_file(model: AbstractRecommenderSystem, path_to_file: str) -> None:
    """
    Method to save model to file in pickle format

    Parameters
    ----------
    model: AbstractRecommenderSystem
        Recommender system model
    path_to_file: str
        Path to file where model will be saved

    Raises
    ------
    TypeError
        If parameters don't have needed format
    """

    if not isinstance(model, AbstractRecommenderSystem):
        raise TypeError('Model should have recommender system format')

    if type(path_to_file) != str:
        raise TypeError('Path should have string format')

    with open(path_to_file, 'wb') as file:
        pickle.dump(model, file)


def get_model_from_file(path_to_file: str) -> AbstractRecommenderSystem:
    """
    Method to get model from file in pickle format

    Parameters
    ----------
    path_to_file: str
        Path to file where model

    Raises
    ------
    ValueError
        If the file does not contain a model
    TypeError
        If parameters don't have string format

    Returns
    -------
    model: AbstractRecommenderSystem
    """

    if type(path_to_file) != str:
        raise TypeError('Path should have string format')

    with open(path_to_file, 'rb') as file:
        model = pickle.load(file)

    if not isinstance(model, AbstractRecommenderSystem):
        raise ValueError('There is no model in this file')

    return model
