import typing as tp

from utilspie.collectionsutils import frozendict

from recommender_system_api.backend.work_with_database import *
from recommender_system_library.extra_functions.work_with_models import MODELS_NAMES
from recommender_system_library.models.abstract import AbstractRecommenderSystem
from recommender_system_library.models import *


__all__ = [
    'create_model',
    'save_parameters',
    'get_parameters',
    'save_model',
    'get_model',
    'delete_model'
]


__MODELS_CLASSES = tuple([
    memory_based_models.UserBasedModel, memory_based_models.ItemBasedModel,
    latent_factor_models.AlternatingLeastSquaresModel, latent_factor_models.HierarchicalAlternatingLeastSquaresModel,
    latent_factor_models.StochasticLatentFactorModel, latent_factor_models.SingularValueDecompositionModel,
    implicit_models.ImplicitStochasticLatentFactorModel
])


__MODELS_DICT: tp.Dict[str, AbstractRecommenderSystem.__class__] = frozendict(zip(MODELS_NAMES, __MODELS_CLASSES))


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

    Returns
    -------
    model: AbstractRecommenderSystem
    """

    if type_model not in MODELS_NAMES:
        raise ValueError('There is no such model')

    return __MODELS_DICT[type_model](**parameters)


def save_parameters(user_id: int, system_id: int, parameters: tp.Dict[str, tp.Any]) -> None:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    save_parameters_to_file(system_id, parameters)


def get_parameters(user_id: int, system_id: int) -> tp.Dict[str, tp.Any]:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    return get_parameters_from_file(system_id)


def save_model(user_id: int, system_id: tp.Optional[int], model: AbstractRecommenderSystem, is_clear=False) -> int:

    if system_id is None:
        system_id = insert_new_model_into_table(user_id)

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    if not check_path_exist(get_path_to_folder_with_models()):
        create_folder(get_path_to_folder_with_models())

    if not check_path_exist(get_path_to_folder_with_model(system_id)):
        create_folder(get_path_to_folder_with_model(system_id))

    if not is_clear and check_path_exist(get_path_to_first_model(system_id)):
        first_model = get_model_from_file(system_id)
        save_model_to_file(system_id, model, is_second=first_model.is_trained)
    else:
        save_model_to_file(system_id, model)
        delete_second_model(system_id)

    if is_clear and check_path_exist(get_path_to_second_model(system_id)):
        delete_second_model(system_id)

    return system_id


def get_model(user_id: int, system_id: int, is_train=False) -> AbstractRecommenderSystem:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    if is_train and check_path_exist(get_path_to_second_model(system_id)):
        return get_model_from_file(system_id, is_second=True)

    return get_model_from_file(system_id)


def delete_model(user_id: int, system_id: int) -> None:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    delete_model_folder(system_id)
    delete_model_from_table(user_id, system_id)
