import typing as tp
from abc import ABC


class DebugInterface(ABC):

    def __init__(self, error_functional):
        self.__error_functional = error_functional
        self.__debug_information: tp.Optional[tp.List[float]] = None  # array with errors on each epoch

    def __update_debug_information(self, is_debug: bool) -> None:
        self.__debug_information = [] if is_debug else None

    def __set_debug_information(self, is_debug: bool, *args) -> None:
        if is_debug:
            # calculate error functionality
            self.__debug_information.append(self.__error_functional(*args))

    def get_debug_information(self) -> tp.List[float]:
        """
        Method for getting a list of functionality errors that were optimized during training

        Returns
        -------
        list of functionality errors: list[float]
        """

        if self.__debug_information is None:
            raise AttributeError("No debug information because there was is_debug = False during training")
        else:
            return self.__debug_information

