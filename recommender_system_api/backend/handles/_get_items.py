import typing as tp

import numpy as np

from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *


__all__ = ['get_list_of_items_from_recommender_system']


def get_list_of_items_from_recommender_system(system_id: int, data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:

    user_index, items_count = split_data(data, [KEY_USER, KEY_ITEMS_COUNT])
    model = get_model(system_id)
    result = model.predict(user_index, items_count)
    return {KEY_RESULT: [int(item_index) for item_index in result]}
