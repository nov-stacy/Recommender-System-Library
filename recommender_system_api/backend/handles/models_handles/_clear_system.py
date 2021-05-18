from backend.checkers import *
from backend.handles._settings import *
from backend.work_with_models import *


__all__ = ['clear_recommender_system']


def clear_recommender_system(user_id: int, system_id: int) -> None:
    """
    Method for clearing the model (returning it to the pre-training state)

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

    # getting parameters for cleaning up
    data = get_parameters(user_id, system_id)
    check_dictionary_with_str_keys(data)

    type_model, params = split_data(data, [KEY_TYPE, KEY_PARAMS])
    check_format_of_str(type_model)
    check_dictionary_with_str_keys(params)

    model = create_model(type_model, params)  # creating a new clean model
    delete_training_model(user_id, system_id)  # deleting training the old model
    save_model(user_id, system_id, model, is_clear=True)  # saving a new model
