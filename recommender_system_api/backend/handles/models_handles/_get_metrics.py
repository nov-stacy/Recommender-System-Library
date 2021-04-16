import typing as tp

from recommender_system_api.backend.checkers import *
from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *
from recommender_system_library.extra_functions.work_with_predict_data import calculate_predicted_items
from recommender_system_library.metrics import *


__all__ = ['get_metric_for_recommender_system']


__metrics_functions: tp.Dict[str, tp.Callable] = {
    'precision': precision_k,
    'recall': recall_k
}

__predict_parameter_names: tp.Tuple[str, str] = (
    'k_items',
    'barrier_value'
)


def get_metric_for_recommender_system(user_id: int, system_id: int, metric_name: str,
                                      data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """
    Method for getting metrics from saved models

    Parameters
    ----------
    user_id: int
        Id of the user who uses the system
    metric_name: str
        Name of metric to get
    system_id: int
        Id of the system that needs to be changed
    data: dictionary
        Parameters that the user sent to the system

    Returns
    -------
    Value of metric: dictionary
    """

    # check all ids on integers and positivity
    check_format_of_positive_integer(user_id)
    check_format_of_positive_integer(system_id)

    check_dictionary_with_str_keys(data)  # check params

    # get parameters from request
    keys = [KEY_TEST_DATA, KEY_USERS, KEY_PREDICT_PARAMETER_NAME, KEY_PREDICT_PARAMETER_VALUE]
    matrix_data, users_list, predict_param_name, predict_param_value = split_data(data, keys)

    # check all parameters
    check_format_of_str(matrix_data)
    check_format_of_list_with_not_negative_integers(users_list)
    check_format_of_str(predict_param_name)

    # get data to test
    matrix = get_data(matrix_data)

    # check metric parameter
    if metric_name not in __metrics_functions or predict_param_name not in __predict_parameter_names:
        raise ValueError

    model = get_model(user_id, system_id)  # getting a saved model

    # if need to get a metric for the first k items
    if predict_param_name == __predict_parameter_names[0]:

        y_predict = [
            model.predict(user_index, predict_param_value) for user_index in users_list
        ]

        y_true = [
            calculate_predicted_items(matrix.getrow(user_index).toarray()[0], k_items=predict_param_value)
            for user_index in users_list
        ]

    # if need to get a metric for barrier value
    else:

        y_predict = [
            calculate_predicted_items(model.predict_ratings(user_index), barrier_value=predict_param_value)
            for user_index in users_list
        ]

        y_true = [
            calculate_predicted_items(matrix.getrow(user_index).toarray()[0], barrier_value=predict_param_value)
            for user_index in users_list
        ]

    return {KEY_RESULT: __metrics_functions[metric_name](y_true, y_predict)}
