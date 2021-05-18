import typing as tp

from backend.checkers import *
from backend.handles._settings import *
from backend.work_with_models import *


__all__ = ['change_recommender_system']


def change_recommender_system(user_id: int, system_id: int, data: tp.Dict[str, tp.Any]) -> None:
    """
    Method for changing system parameters

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

    # get old data from database
    old_data = get_parameters(user_id, system_id)
    check_dictionary_with_str_keys(old_data)
    type_model, old_params = split_data(old_data, [KEY_TYPE, KEY_PARAMS])
    check_format_of_str(type_model)
    check_dictionary_with_str_keys(old_params)

    # get new data from request
    params = split_data(data, [KEY_PARAMS])[0]
    check_dictionary_with_str_keys(params)

    if params == old_params:  # if the parameters are the same, then the request does not make sense
        raise AttributeError('The parameters are the same. Use clear handle')

    model = create_model(type_model, params)  # creating a model
    delete_training_model(user_id, system_id)  # deleting training the old model

    # saving a new model
    save_model(user_id, system_id, model)
    save_parameters(user_id, system_id, merge_data([KEY_TYPE, KEY_PARAMS], [type_model, params]))
