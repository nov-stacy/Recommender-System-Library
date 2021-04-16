import typing as tp

from recommender_system_api.backend.handles._settings import KEY_TOKEN
from recommender_system_api.backend.work_with_database import insert_new_user


__all__ = ['registration_user']


def registration_user() -> tp.Dict[str, tp.Any]:

    return {KEY_TOKEN: insert_new_user()}
