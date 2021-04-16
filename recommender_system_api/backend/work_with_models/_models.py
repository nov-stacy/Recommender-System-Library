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


def save_parameters(system_id: int, user_id: int, parameters: tp.Dict[str, tp.Any]) -> None:

    if not check_model(system_id, user_id):
        raise ValueError

    save_parameters_to_file(system_id, parameters)


def get_parameters(system_id: int, user_id: int) -> tp.Dict[str, tp.Any]:

    if not check_model(system_id, user_id):
        raise ValueError

    return get_parameters_from_file(system_id)


def save_model(system_id: tp.Optional[int], user_id: int, model: AbstractRecommenderSystem, is_clear=False) -> int:

    if system_id is None:
        system_id = insert_new_model(user_id)

    if not check_model(system_id, user_id):
        raise ValueError

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


def get_model(system_id: int, user_id: int) -> AbstractRecommenderSystem:

    if not check_model(system_id, user_id):
        raise ValueError

    return get_model_from_file(system_id)


def delete_model(system_id: int, user_id: int) -> None:

    if not check_model(system_id, user_id):
        raise ValueError

    delete_model_folder(system_id)
    delete_model(system_id, user_id)
