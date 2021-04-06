import typing as tp

from recommender_system_api.backend.work_with_database._settings import KEY_TYPE, KEY_PARAMS


def split_data(data: tp.Dict[str, tp.Any]) -> tp.Tuple[str, tp.Dict[str, tp.Any]]:

    if set(data.keys()) != {KEY_TYPE, KEY_PARAMS}:
        raise ValueError

    return data[KEY_TYPE], data[KEY_PARAMS]


def merge_data(type_model: str, params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:

    return {KEY_TYPE: type_model, KEY_PARAMS: params}
