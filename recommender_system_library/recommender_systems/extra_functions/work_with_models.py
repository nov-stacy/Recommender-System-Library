import pickle

from recommender_systems.models.abstract import AbstractRecommenderSystem


__all__ = [
    'MODELS_NAMES',
    'save_model_to_file',
    'get_model_from_file'
]


MODELS_NAMES = tuple([
    'user_based_model', 'item_based_model', 'latent_factor_als_model', 'latent_factor_hals_model',
    'latent_factor_sgd_model', 'latent_factor_svd_model', 'implicit_sgd_model'
])


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
