from recommender_system_api.backend.checkers import check_format_of_positive_integer
from recommender_system_api.backend.work_with_models import *


__all__ = ['delete_recommender_system']


def delete_recommender_system(user_id: int, system_id: int) -> None:
    """
    Method for deleting a model from the server

    Parameters
    ----------
    user_id: int
        Id of the user who uses the system
    system_id: int
        Id of the system that needs to be changed
    """

    # check all ids on integers and positivity
    check_format_of_positive_integer(user_id)
    check_format_of_positive_integer(system_id)

    delete_training_model(user_id, system_id)  # deleting training the old model
    delete_model(user_id, system_id)
