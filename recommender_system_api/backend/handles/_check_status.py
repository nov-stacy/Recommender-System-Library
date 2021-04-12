import typing as tp

from recommender_system_api.backend.work_with_models import check_status_thread


def check_status(system_id: int) -> tp.Dict[str, tp.Any]:
    return {'status': check_status_thread(system_id)}
