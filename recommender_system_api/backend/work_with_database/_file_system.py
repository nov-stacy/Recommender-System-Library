import os
import shutil

from recommender_system_api.backend.work_with_database._settings import PATH_TO_DATABASE, MODELS_FOLDER


def check_path_exist(path) -> bool:
    return os.path.exists(path)


def get_path_to_folder_with_models() -> str:
    return f'{PATH_TO_DATABASE}/{MODELS_FOLDER}'


def get_path_to_folder_with_model(system_id: int) -> str:
    return f'{PATH_TO_DATABASE}/{MODELS_FOLDER}/{system_id}'


def create_folder(path_to_folder: str) -> None:
    os.mkdir(path_to_folder)


def delete_model_folder(system_id: int) -> None:
    shutil.rmtree(get_path_to_folder_with_model(system_id), ignore_errors=True)
