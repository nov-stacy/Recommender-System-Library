import typing as tp

import numpy as np

from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *


__all__ = ['get_list_of_ratings_from_recommender_system']


def get_list_of_ratings_from_recommender_system(system_id: int, data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:

    user_index = split_data(data, [KEY_USER])[0]
    model = get_model(system_id)
    return {KEY_RESULT: list(model.predict_ratings(user_index))}
