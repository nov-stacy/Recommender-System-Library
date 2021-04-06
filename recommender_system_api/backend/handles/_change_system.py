import typing as tp

from recommender_system_api.backend.work_with_database import get_parameters_from_file, split_data, merge_data
from recommender_system_api.backend.work_with_database import save_model_to_file, save_parameters_to_file
from recommender_system_library.extra_functions.work_with_models import create_model


def change_recommender_system(system_id: int, params: tp.Dict[str, tp.Any]) -> None:

    old_data = get_parameters_from_file(system_id)
    type_model, old_params = split_data(old_data)

    if params == old_params:
        return

    new_params = old_params.copy()
    new_params.update(params)

    model = create_model(type_model, new_params)
    save_model_to_file(system_id, model)
    save_parameters_to_file(system_id, merge_data(type_model, new_params))
