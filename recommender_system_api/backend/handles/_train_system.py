import time
import typing as tp

from recommender_system_api.backend.handles._settings import *
from recommender_system_api.backend.work_with_models import *


__all__ = ['train_recommender_system']


def train_recommender_system(system_id: int, data: tp.Dict[str, tp.Any]) -> None:

    parameters, train_matrix_str = split_data(data, [KEY_PARAMS, KEY_TRAIN_DATA])
    model = get_model(system_id)

    train_matrix = get_data(train_matrix_str)

    thread = TrainThread(system_id, model, train_matrix, parameters)
    thread.setDaemon(True)
    thread.start()

    time.sleep(1)

    if not add_thread(system_id, thread):
        raise ValueError
