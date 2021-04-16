from recommender_system_api.backend.work_with_models import delete_model, delete_thread

__all__ = ['delete_recommender_system']


def delete_recommender_system(user_id: int, system_id: int) -> None:
    delete_thread(system_id)
    delete_model(system_id, user_id)
