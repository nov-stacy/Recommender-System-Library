import typing as tp

import numpy as np


__all__ = [
    'check_format_of_integer', 'check_format_of_positive_integer', 'check_format_of_str',
    'check_format_of_list_with_not_negative_integers',
    'check_dictionary_with_str_keys'
]


def check_format_of_integer(value: int) -> None:
    """
    Method to check value for integer

    Parameters
    ----------
    value: int
    """

    if type(value) not in [int, np.int64]:
        raise TypeError('Value should be int')


def check_format_of_positive_integer(value: int) -> None:
    """
    Method to check value for positive integer

    Parameters
    ----------
    value: int
    """

    check_format_of_integer(value)

    if value <= 0:
        raise ValueError('Value should be positive')


def check_format_of_str(value: str) -> None:
    """
    Method to check value for string

    Parameters
    ----------
    value: str
    """

    if type(value) != str:
        raise TypeError('Value should be str')


def check_format_of_list_with_not_negative_integers(value: tp.List[int]) -> None:
    """
    Method to check value for list

    Parameters
    ----------
    value: list
    """

    if type(value) != list:
        raise TypeError('Value should be list')

    for element in value:
        if type(element) not in [int, np.int64]:
            raise TypeError('Values in list should be integers')
        if element < 0:
            raise ValueError('Values in list should be not negative')


def check_dictionary_with_str_keys(value: tp.Dict[str, tp.Any]) -> None:
    """
    Method to check value for dictionary

    Parameters
    ----------
    value: dictionary
    """

    if type(value) != dict:
        raise TypeError('Value should be dictionary')

    for element in value:
        if type(element) != str:
            raise TypeError('Keys in dictionary should have string format')
