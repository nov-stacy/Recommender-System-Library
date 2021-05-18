import typing as tp

from backend.checkers import *
from backend.handles._settings import KEY_STATUS
from backend.work_with_models import check_status_of_system


__all__ = ['check_status_of_recommender_system']


def check_status_of_recommender_system(user_id: int, system_id: int) -> tp.Dict[str, tp.Any]:
    """
    Method for checking the model status: learning, not learning, error during training

    Parameters
    ----------
    user_id: int
        Id of the user who uses the system
    system_id: int
        Id of the system that needs to be changed

    Returns
    -------
    status of the system: dictionary
    """

    # check all ids on integers and positivity
    check_format_of_positive_integer(user_id)
    check_format_of_positive_integer(system_id)

    return {KEY_STATUS: check_status_of_system(user_id, system_id)}
