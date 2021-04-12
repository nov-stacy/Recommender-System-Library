import pickle
import typing as tp

import os
import shutil

from recommender_system_api.backend.work_with_database._settings import *

from recommender_system_library.extra_functions.work_with_models import save_model_to_file as _save_model
from recommender_system_library.extra_functions.work_with_models import get_model_from_file as _get_model_from_file
from recommender_system_library.models.abstract import AbstractRecommenderSystem


def check_path_exist(path) -> bool:
    return os.path.exists(path)


def get_path_to_folder_with_models() -> str:
    return f'{PATH_TO_DATABASE}/{MODELS_FOLDER}'


def get_path_to_folder_with_model(system_id: int) -> str:
    return f'{PATH_TO_DATABASE}/{MODELS_FOLDER}/{system_id}'


def get_path_to_first_model(system_id: int) -> str:
    return f'{get_path_to_folder_with_model(system_id)}/{MODEL_FILE}'


def get_path_to_second_model(system_id: int) -> str:
    return f'{get_path_to_folder_with_model(system_id)}/{SECOND_MODEL_FILE}'


def create_folder(path_to_folder: str) -> None:
    os.mkdir(path_to_folder)


def delete_model_folder(system_id: int) -> None:
    shutil.rmtree(get_path_to_folder_with_model(system_id), ignore_errors=True)


def delete_second_model(system_id: int) -> None:
    os.remove(get_path_to_second_model(system_id))


def save_model_to_file(system_id: int, model: AbstractRecommenderSystem, is_second=False) -> None:
    if is_second:
        _save_model(model, get_path_to_second_model(system_id))
    else:
        _save_model(model, get_path_to_first_model(system_id))


def get_model_from_file(system_id: int, is_second=False) -> AbstractRecommenderSystem:
    if is_second:
        return _get_model_from_file(get_path_to_second_model(system_id))
    else:
        return _get_model_from_file(get_path_to_first_model(system_id))


def save_parameters_to_file(system_id: int, parameters: tp.Dict[str, tp.Any]) -> None:
    with open(f'{get_path_to_folder_with_model(system_id)}/{PARAMETERS_FILE}', 'wb') as file:
        pickle.dump(parameters, file)


def get_parameters_from_file(system_id: int) -> tp.Dict[str, tp.Any]:
    with open(f'{get_path_to_folder_with_model(system_id)}/{PARAMETERS_FILE}', 'rb') as file:
        return pickle.load(file)


__all__ = [
    'check_path_exist',
    'get_path_to_folder_with_models',
    'get_path_to_folder_with_model',
    'get_path_to_first_model',
    'get_path_to_second_model',
    'create_folder',
    'delete_model_folder',
    'delete_second_model',
    'save_model_to_file',
    'get_model_from_file',
    'save_parameters_to_file',
    'get_parameters_from_file'
]
