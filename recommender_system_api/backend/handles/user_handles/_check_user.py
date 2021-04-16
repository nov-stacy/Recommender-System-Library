import typing as tp

from recommender_system_api.backend.handles._settings import KEY_TOKEN
from recommender_system_api.backend.work_with_database import check_user


__all__ = ['check_user_token']


def check_user_token(headers: tp.Dict[str, tp.Any]) -> int:

    token = headers.get(KEY_TOKEN)
    user_id = check_user(token)

    if user_id is None:
        pass

    return user_id
