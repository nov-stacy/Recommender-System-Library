import typing as tp

from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *
from recommender_system_library.extra_functions.work_with_models import create_model


__all__ = ['change_recommender_system']


def change_recommender_system(user_id: int, system_id: int, data: tp.Dict[str, tp.Any]) -> None:

    old_data = get_parameters(system_id, user_id)
    type_model, old_params = split_data(old_data, [KEY_TYPE, KEY_PARAMS])
    params = split_data(data, [KEY_PARAMS])[0]

    if params == old_params:
        raise AttributeError

    model = create_model(type_model, params)

    delete_thread(system_id)

    save_model(system_id, model, user_id)
    save_parameters(system_id, user_id, merge_data([KEY_TYPE, KEY_PARAMS], [type_model, params]))
