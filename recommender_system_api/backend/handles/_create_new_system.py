import typing as tp

from recommender_system_api.backend.work_with_database import split_data
from recommender_system_api.backend.work_with_database import save_new_model_to_file, save_parameters_to_file
from recommender_system_library.extra_functions.work_with_models import create_model


def create_recommender_system(data: tp.Dict[str, tp.Any]) -> int:

    type_model, params = split_data(data)
    model = create_model(type_model, params)
    system_id = save_new_model_to_file(model)
    save_parameters_to_file(system_id, data)

    return system_id
