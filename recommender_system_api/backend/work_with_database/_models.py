import typing as tp

import pickle

from recommender_system_api.backend.work_with_database._file_system import check_path_exist, create_folder
from recommender_system_api.backend.work_with_database._file_system import get_path_to_folder_with_models
from recommender_system_api.backend.work_with_database._file_system import get_path_to_folder_with_model
from recommender_system_api.backend.work_with_database._file_system import delete_model_folder

from recommender_system_api.backend.work_with_database._settings import PARAMETERS_FILE, MODEL_FILE
from recommender_system_api.backend.work_with_database._sql import check_system, insert_new_system

from recommender_system_library.models.abstract import AbstractRecommenderSystem
from recommender_system_library.extra_functions.work_with_models import save_model_to_file as _save_model
from recommender_system_library.extra_functions.work_with_models import get_model_from_file as _get_model_from_file


def save_parameters_to_file(system_id: int, parameters: tp.Dict[str, tp.Any]) -> None:

    if not check_system(system_id):
        raise ValueError

    with open(f'{get_path_to_folder_with_model(system_id)}/{PARAMETERS_FILE}', 'wb') as file:
        pickle.dump(parameters, file)


def get_parameters_from_file(system_id: int) -> tp.Dict[str, tp.Any]:

    if not check_system(system_id):
        raise ValueError

    with open(f'{get_path_to_folder_with_model(system_id)}/{PARAMETERS_FILE}', 'rb') as file:
        return pickle.load(file)


def save_model_to_file(system_id, model: AbstractRecommenderSystem) -> None:

    if not check_system(system_id):
        raise ValueError

    if not check_path_exist(get_path_to_folder_with_models()):
        create_folder(get_path_to_folder_with_models())

    _save_model(model, f'{get_path_to_folder_with_model(system_id)}/{MODEL_FILE}')

    return system_id


def save_new_model_to_file(model: AbstractRecommenderSystem) -> int:

    system_id = insert_new_system()

    create_folder(system_id)

    _save_model(model, f'{get_path_to_folder_with_model(system_id)}/{MODEL_FILE}')

    return system_id


def get_model_from_file(system_id: int) -> AbstractRecommenderSystem:

    if not check_system(system_id):
        raise ValueError

    return _get_model_from_file(f'{get_path_to_folder_with_model(system_id)}/{MODEL_FILE}')


def delete_model_file(system_id: int) -> None:

    if not check_system(system_id):
        raise ValueError

    delete_model_folder(system_id)
