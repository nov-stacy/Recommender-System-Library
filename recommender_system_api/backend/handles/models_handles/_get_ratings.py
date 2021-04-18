import typing as tp

from recommender_system_api.backend.checkers import *
from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *


__all__ = ['get_list_of_ratings_from_recommender_system']


def get_list_of_ratings_from_recommender_system(user_id: int, system_id: int,
                                                data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """
    Method for getting the ratings for all items from the saved model

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
    list of ratings: dictionary
    """

    # check all ids on integers and positivity
    check_format_of_positive_integer(user_id)
    check_format_of_positive_integer(system_id)

    check_dictionary_with_str_keys(data)  # check params

    user_index = split_data(data, [KEY_USER])[0]  # get data from request
    check_format_of_integer(user_index)  # check index of user
    model = get_model(user_id, system_id)  # getting a saved model

    return {KEY_RESULT: list(model.predict_ratings(user_index))}
