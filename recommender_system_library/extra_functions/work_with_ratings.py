import typing as tp

import numpy as np


__all__ = ['calculate_predicted_items']


def calculate_predicted_items(ratings: np.ndarray,
                              k_items: tp.Optional[int] = None,
                              barrier_value: tp.Optional[float] = None) -> np.ndarray:
    """
    Method for getting a ranked list of items that are recommended to the ratings

    Parameters
    ----------
    ratings: numpy array
        Array of ratings which predicted to user
    k_items: [int, None]
        Number of items to recommend
    barrier_value: [float, None]
        Value of rating greater than or equal to which to recommend

    Raises
    ------
    TypeError
        If parameters don't have needed format
    ValueError
        If both of optional parameters or None or not None
        If ratings is not 1D array

    Returns
    -------
    array of indices (sorted by rating values): numpy array
    """

    if type(ratings) != np.ndarray:
        raise TypeError('Ratings should have numpy array format')

    if ratings.dtype not in [np.int, np.float64]:
        raise TypeError('Arrays should store indices')

    if sum(value is not None for value in [k_items, barrier_value]) != 1:
        raise ValueError('You should assign only one of the arguments non-None')

    if not(type(k_items) == int or type(barrier_value) in [np.float64, float, int]):
        raise TypeError('k_items should have int format or barrier_value should have float format')

    if len(ratings.shape) != 1:
        raise ValueError('Ratings need to be 1D array')

    # indexes sorted in descending order of rating value and base mask
    sort_indices = ratings.argsort()[::-1]
    mask = np.array([True] * ratings.shape[0])

    # if need to take the first k ratings
    if k_items is not None:
        if k_items < 0 or k_items > ratings.shape[0]:
            raise ValueError('Count of items should be not negative value and not bigger than count of ratings')
        sort_indices = sort_indices[:k_items]

    # if need to take all ratings that not lower than value
    if barrier_value is not None:
        mask = ratings >= barrier_value

    return sort_indices[mask[sort_indices]]
