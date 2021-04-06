import pickle
import typing as tp

from utilspie.collectionsutils import frozendict

from recommender_system_library.models.abstract import AbstractRecommenderSystem
from recommender_system_library.models import *


__models_dict: tp.Dict[str, AbstractRecommenderSystem.__class__] = frozendict({
    'user_based_model': memory_based_models.UserBasedModel,
    'item_based_model': memory_based_models.ItemBasedModel,
    'latent_factor_als_model': latent_factor_models.AlternatingLeastSquaresModel,
    'latent_factor_hals_model': latent_factor_models.HierarchicalAlternatingLeastSquaresModel,
    'latent_factor_sgd_model': latent_factor_models.StochasticLatentFactorModel,
    'latent_factor_svd_model': latent_factor_models.SingularValueDecompositionModel,
    'implicit_sgd_model': implicit_models.StochasticImplicitLatentFactorModel
})


models_types = list(__models_dict.keys())


def create_model(type_model: str, parameters: tp.Dict[str, tp.Any]) -> AbstractRecommenderSystem:

    if type_model not in models_types:
        raise KeyError('There is no such model')

    return __models_dict[type_model](**parameters)


def save_model_to_file(model: AbstractRecommenderSystem, path_to_file: str) -> None:

    with open(path_to_file, 'wb') as file:
        pickle.dump(model, file)


def get_model_from_file(path_to_file: str) -> AbstractRecommenderSystem:

    with open(path_to_file, 'rb') as file:
        return pickle.load(file)
