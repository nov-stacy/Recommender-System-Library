import typing as tp

from recommender_system_api.backend.checkers import *
from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *


__all__ = ['create_recommender_system']


def create_recommender_system(user_id: int, data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """
    Method for creating a new model

    Parameters
    ----------
    user_id: int
        Id of the user who uses the system
    data: dictionary
        Parameters that the user sent to the system

    Returns
    -------
    id of new system: dictionary
    """

    check_format_of_positive_integer(user_id)  # check all ids on integers and positivity
    check_dictionary_with_str_keys(data)  # check params

    type_model, params = split_data(data, [KEY_TYPE, KEY_PARAMS])  # get new data from request
    check_format_of_str(type_model)
    check_dictionary_with_str_keys(params)
    model = create_model(type_model, params)  # creating a model

    # saving a new model
    system_id = save_model(user_id, None, model)
    save_parameters(user_id, system_id, data)

    return {KEY_ID: system_id}
