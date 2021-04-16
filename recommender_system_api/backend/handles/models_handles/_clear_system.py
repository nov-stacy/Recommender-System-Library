from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *
from recommender_system_library.extra_functions.work_with_models import create_model


__all__ = ['clear_recommender_system']


def clear_recommender_system(user_id: int, system_id: int) -> None:

    data = get_parameters(system_id, user_id)
    type_model, params = split_data(data, [KEY_TYPE, KEY_PARAMS])

    model = create_model(type_model, params)

    delete_thread(system_id)

    save_model(system_id, user_id, model, is_clear=True)
