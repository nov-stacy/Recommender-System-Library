import numpy as np
import typing as tp


def calculate_barrier_value(ratings: np.ndarray, probability: float) -> float:
    """
    Method to calculate value which the rating exceeds with probability

    Parameters
    ----------
    ratings: numpy array
        Array of ratings which predicted to user
    probability: float
        Float value between [0, 1]

    Returns
    -------
    barrier value: float
    """

    if len(ratings.shape) != 1:
        raise ValueError('Ratings need to be 1D array')

    if probability > 1 or probability < 0:
        raise ValueError('probability should be in [0, 1]')

    return np.quantile(ratings, 1 - probability)


def calculate_issue_ranked_list(ratings: np.ndarray,
                                k_items: tp.Optional[int] = None,
                                barrier_value: tp.Optional[float] = None) -> np.ndarray:
    """
    Method for getting a ranked list of items that are recommended to the ratings

    Parameters
    ----------
    ratings: numpy arrray
        Array of ratings which predicted to user
    k_items: [int, None]
        Number of items to recommend
    barrier_value: [float, None]
        Value of rating greater than or equal to which to recommend

    Returns
    -------
    array of indices (sorted by rating values): numpy array
    """

    if len(ratings.shape) != 1:
        raise ValueError('Ratings need to be 1D array')

    if sum(value is not None for value in [k_items, barrier_value]) != 1:
        raise ValueError('You should assign only one of the arguments non-None')

    # indexes sorted in descending order of rating value and base mask
    sort_indices = ratings.argsort()[::-1]
    mask = np.array([True] * ratings.shape[0])

    # if need to take the first k ratings
    if k_items:
        if k_items < 0 or k_items > ratings.shape[0]:
            raise ValueError('Count of items should be not negative value and not bigger than count of ratings')
        sort_indices = sort_indices[:k_items]

    # if need to take all ratings that not lower than value
    if barrier_value:
        mask = ratings >= barrier_value

    return sort_indices[mask[sort_indices]]