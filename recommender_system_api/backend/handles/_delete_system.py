from recommender_system_api.backend.work_with_database import delete_system, delete_model_file


def delete_recommender_system(system_id: int):

    delete_system(system_id)
    delete_model_file(system_id)
