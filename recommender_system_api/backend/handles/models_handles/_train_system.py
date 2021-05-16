import time
import typing as tp

from recommender_system_api.backend.checkers import *
from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *


__all__ = ['train_recommender_system']


def train_recommender_system(user_id: int, system_id: int, data: tp.Dict[str, tp.Any]) -> None:
    """
    Method for starting model training

    Parameters
    ----------
    user_id: int
        Id of the user who uses the system
    system_id: int
        Id of the system that needs to be changed
    data: dictionary
        Parameters that the user sent to the system
    """

    # check all ids on integers and positivity
    check_format_of_positive_integer(user_id)
    check_format_of_positive_integer(system_id)

    check_dictionary_with_str_keys(data)  # check params

    # get parameters from request
    train_matrix_str, parameters = split_data(data, [KEY_TRAIN_DATA, KEY_PARAMS])
    check_format_of_str(train_matrix_str)
    check_dictionary_with_str_keys(parameters)

    model = get_model(user_id, system_id, is_train=True)  # getting a saved model

    train_matrix = get_data(train_matrix_str)

    thread = TrainThread(user_id, system_id, model, train_matrix, parameters)
    thread.setDaemon(True)
    thread.start()
    time.sleep(1)

    add_model_to_train_system(user_id, system_id, thread)
