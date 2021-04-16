import typing as tp

from recommender_system_api.backend.checkers import *
from recommender_system_api.backend.handles._settings import KEY_TOKEN
from recommender_system_api.backend.work_with_database import check_user
from recommender_system_api.backend.work_with_models import split_data


__all__ = ['check_user_token']


def check_user_token(headers: tp.Dict[str, tp.Any]) -> int:
    """
    Method for getting user id for token

    Parameters
    ----------
    headers: dictionary
        Parameters that the user sent to the system

    Returns
    -------
    id of new user: int
    """

    # check all ids on integers and positivity
    check_dictionary_with_str_keys(headers)

    # get user id with token
    token = split_data(headers, [KEY_TOKEN])[0]
    user_id = check_user(token)

    return user_id
