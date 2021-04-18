import typing as tp

from recommender_system_api.backend.work_with_database import *
from recommender_system_library.models.abstract import AbstractRecommenderSystem


__all__ = [
    'save_parameters',
    'get_parameters',
    'save_model',
    'get_model',
    'delete_model'
]


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

    if is_clear and check_path_exist(get_path_to_second_model(system_id)):
        delete_second_model(system_id)

    return system_id


def get_model(user_id: int, system_id: int) -> AbstractRecommenderSystem:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    return get_model_from_file(system_id)


def delete_model(user_id: int, system_id: int) -> None:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    delete_model_folder(system_id)
    delete_model_from_table(user_id, system_id)
