from recommender_system_api.backend.work_with_database import get_parameters_from_file, split_data, save_model_to_file
from recommender_system_library.extra_functions.work_with_models import create_model


def clear_recommender_system(system_id: int) -> None:

    data = get_parameters_from_file(system_id)
    type_model, params = split_data(data)

    model = create_model(type_model, params)
    save_model_to_file(system_id, model)
