import ctypes
import inspect
import threading

import typing as tp

from scipy import sparse

from recommender_system_api.backend.work_with_models import *
from recommender_system_library.models.abstract import AbstractRecommenderSystem


__all__ = [
    'ERROR_STATUS',
    'NOT_TRAIN_STATUS',
    'TRAIN_STATUS',
    'TrainThread',
    'add_thread',
    'delete_thread',
    'check_status_thread'
]

ERROR_STATUS = 'ERROR'
NOT_TRAIN_STATUS = 'NOT TRAINING'
TRAIN_STATUS = 'TRAINING'


def _async_raise(tid, exc_type):
    if not inspect.isclass(exc_type):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exc_type))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class TrainThread(threading.Thread):

    def __init__(self, system_id: int, user_id: int, model: AbstractRecommenderSystem, data: sparse.coo_matrix,
                 train_parameters: tp.Dict[str, tp.Any]) -> None:
        super().__init__()
        self.__system_id = system_id
        self.__user_id = user_id
        self.__model = model
        self.__data = data
        self.__train_parameters = train_parameters
        self.is_training = True
        self.is_error = False

    def _get_my_tid(self):
        if not self.is_alive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, t_obj in threading._active.items():
            if t_obj is self:
                self._thread_id = tid
                return tid

        raise AssertionError("could not determine the thread's id")

    def run(self) -> None:
        try:
            if self.__model.is_trained:
                self.__model.refit(self.__data, **self.__train_parameters)
            else:
                self.__model.fit(self.__data, **self.__train_parameters)
            save_model(self.__system_id, self.__user_id, self.__model, is_clear=True)
        except Exception:
            self.is_error = True
        finally:
            self.is_training = False

    def raise_exc(self, exc_type):
        _async_raise(self._get_my_tid(), exc_type)

    def terminate(self):
        self.raise_exc(SystemExit)


__training_threads: tp.Dict[int, TrainThread] = dict()


# TODO CHECK MODEL FOR USER ID


def add_thread(system_id: int, thread: TrainThread) -> bool:

    if thread.is_error:
        return False

    __training_threads[system_id] = thread
    return True


def delete_thread(system_id) -> None:

    if system_id not in __training_threads:
        return

    __training_threads[system_id].terminate()
    del __training_threads[system_id]


def check_status_thread(system_id) -> str:

    if system_id not in __training_threads:
        return NOT_TRAIN_STATUS

    if __training_threads[system_id].is_training:
        return TRAIN_STATUS

    status = ERROR_STATUS if __training_threads[system_id].is_error else NOT_TRAIN_STATUS
    delete_thread(system_id)

    return status
