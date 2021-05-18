import typing as tp

from backend.checkers import *
from backend.handles._settings import KEY_TOKEN
from backend.work_with_database import check_user_in_table


__all__ = ['check_user_token', 'AuthTokenError']


class AuthTokenError(Exception):
    def __init__(self, text):
        self.txt = text


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

    # get user id with token
    token = headers.get(KEY_TOKEN)
    user_id = check_user_in_table(token)

    if user_id is None:
        raise AuthTokenError('Token is not valid')

    check_format_of_positive_integer(user_id)

    return user_id
