import ctypes
import inspect
import threading

import typing as tp

from scipy import sparse

from backend.work_with_database import check_model_in_table
from backend.work_with_models import *
from recommender_systems.models.abstract import AbstractRecommenderSystem


__all__ = [
    'ERROR_STATUS',
    'NOT_TRAIN_STATUS',
    'TRAIN_STATUS',
    'TrainThread',
    'add_model_to_train_system',
    'delete_training_model',
    'check_status_of_system'
]

ERROR_STATUS = 'ERROR DURING THE TRAINING'
NOT_TRAIN_STATUS = 'READY'
TRAIN_STATUS = 'TRAINING'


def _async_raise(tid, exc_type):
    if not inspect.isclass(exc_type):
        raise TypeError('Only types can be raised (not instances)')
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exc_type))
    if res == 0:
        raise ValueError('Invalid thread id')
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError('PyThreadState_SetAsyncExc failed')


class TrainThread(threading.Thread):
    """

    """

    def __init__(self, user_id: int, system_id: int, model: AbstractRecommenderSystem, data: sparse.coo_matrix,
                 train_parameters: tp.Dict[str, tp.Any]) -> None:
        super().__init__()
        self._system_id = system_id
        self._user_id = user_id
        self._model = model
        self._data = data
        self._train_parameters = train_parameters
        self.is_training = True
        self.is_error = False

    def _get_my_tid(self):
        if not self.is_alive():
            raise threading.ThreadError('The thread is not active')

        # do we have it cached?
        if hasattr(self, '_thread_id'):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, t_obj in threading._active.items():
            if t_obj is self:
                self._thread_id = tid
                return tid

        raise AssertionError('Could not determine the thread id')

    def run(self) -> None:
        try:
            if self._model.is_trained:
                self._model.refit(self._data, **self._train_parameters)
            else:
                self._model.fit(self._data, **self._train_parameters)
            save_model(self._user_id, self._system_id, self._model, is_clear=True)
        except Exception:
            self.is_error = True
        finally:
            self.is_training = False

    def raise_exc(self, exc_type):
        _async_raise(self._get_my_tid(), exc_type)

    def terminate(self):
        self.raise_exc(SystemExit)


__training_threads: tp.Dict[int, TrainThread] = dict()


def add_model_to_train_system(user_id: int, system_id: int, thread: TrainThread) -> None:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    if thread.is_error:
        raise ValueError('Error in training')

    __training_threads[system_id] = thread


def delete_training_model(user_id: int, system_id: int) -> None:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    if system_id not in __training_threads:
        return

    if __training_threads[system_id].is_alive():
        __training_threads[system_id].terminate()
    del __training_threads[system_id]


def check_status_of_system(user_id: int, system_id: int) -> str:

    if not check_model_in_table(user_id, system_id):
        raise AttributeError('No access to this model')

    if system_id not in __training_threads:
        return NOT_TRAIN_STATUS

    if __training_threads[system_id].is_training:
        return TRAIN_STATUS

    status = ERROR_STATUS if __training_threads[system_id].is_error else NOT_TRAIN_STATUS
    delete_training_model(user_id, system_id)

    return status
