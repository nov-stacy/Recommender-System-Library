import typing as tp

from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *
from recommender_system_library.extra_functions.work_with_models import *


__all__ = ['create_recommender_system']


def create_recommender_system(user_id: int, data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:

    type_model, params = split_data(data, [KEY_TYPE, KEY_PARAMS])
    model = create_model(type_model, params)
    system_id = save_model(None, user_id, model)
    save_parameters(system_id, user_id, data)

    return {KEY_ID: system_id}
