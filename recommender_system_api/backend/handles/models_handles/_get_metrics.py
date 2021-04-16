import typing as tp

import numpy as np

from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *
from recommender_system_library.extra_functions.work_with_predict_data import calculate_predicted_items
from recommender_system_library.metrics import *


__all__ = ['get_metric_for_recommender_system']


__metrics: tp.Dict[str, tp.Callable] = {
    'precision': precision_k,
    'recall': recall_k
}

__predict_parameter_names = (
    'k_items',
    'barrier_value'
)


def get_metric_for_recommender_system(user_id: int, metric_name: str, system_id: int, data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:

    keys = [KEY_TEST_DATA, KEY_USERS, KEY_PREDICT_PARAMETER_NAME, KEY_PREDICT_PARAMETER_VALUE]
    matrix_data, users_list, predict_param_name, predict_param_value = split_data(data, keys)

    matrix = get_data(matrix_data)
    metric = __metrics[metric_name]

    if predict_param_name not in __predict_parameter_names:
        raise ValueError

    model = get_model(system_id, user_id)

    if predict_param_name == __predict_parameter_names[0]:

        y_predict = [
            model.predict(user_index, predict_param_value) for user_index in users_list
        ]

        y_true = [
            calculate_predicted_items(matrix.getrow(user_index).toarray()[0], k_items=predict_param_value)
            for user_index in users_list
        ]

    else:

        y_predict = [
            calculate_predicted_items(model.predict_ratings(user_index), barrier_value=predict_param_value)
            for user_index in users_list
        ]

        y_true = [
            calculate_predicted_items(matrix.getrow(user_index).toarray()[0], barrier_value=predict_param_value)
            for user_index in users_list
        ]

    return {KEY_RESULT: metric(y_true, y_predict)}
