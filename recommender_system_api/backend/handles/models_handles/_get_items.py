import typing as tp

from backend.checkers import *
from backend.handles._settings import *
from backend.work_with_models import *


__all__ = ['get_list_of_items_from_recommender_system']


def get_list_of_items_from_recommender_system(user_id: int, system_id: int,
                                              data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """
    Method for getting the predicted items from the saved model

    Parameters
    ----------
    user_id: int
        Id of the user who uses the system
    system_id: int
        Id of the system that needs to be changed
    data: dictionary
        Parameters that the user sent to the system

    Returns
    -------
    list of items: dictionary
    """

    # check all ids on integers and positivity
    check_format_of_positive_integer(user_id)
    check_format_of_positive_integer(system_id)

    check_dictionary_with_str_keys(data)  # check params

    # getting the user's index
    user_index = split_data(data, [KEY_USER])[0]
    check_format_of_integer(user_index)

    model = get_model(user_id, system_id)  # getting a saved model

    return {KEY_RESULT: [int(item_index) for item_index in model.predict(user_index)]}
