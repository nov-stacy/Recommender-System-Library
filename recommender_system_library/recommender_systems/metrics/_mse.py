import typing as tp
import numpy as np


__all__ = ['mean_square_error']


def _one_mean_square_error(true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
    """
    Method to calculate MSE for one user

    Parameters
    ----------
    true_ratings: numpy array
        Ratings of items, about which it is known that they was liked by the user
    predicted_ratings: numpy array
        Item ratings that were recommended to the user

    Returns
    -------
    MSE for user: float
    """

    if type(true_ratings) != np.ndarray or type(predicted_ratings) != np.ndarray:
        raise TypeError('Ratings need to have numpy array format')

    if true_ratings.shape != predicted_ratings.shape or len(true_ratings.shape) != 1:
        raise ValueError('True and predicted ratings need to be 1D array with same shape')

    if true_ratings.shape[0] and true_ratings.dtype not in [np.float, np.int] or \
            predicted_ratings.shape[0] and predicted_ratings.dtype not in [np.float, np.int]:
        raise TypeError('Arrays should store ratings')

    return np.sum((predicted_ratings - true_ratings) ** 2) / predicted_ratings.shape[0]


def mean_square_error(true_ratings: tp.List[np.ndarray], predicted_ratings: tp.List[np.ndarray]) -> float:
    """
    Method to calculate MSE for all users

    Parameters
    ----------
    true_ratings: array of numpy arrays
        Ratings of items, about which it is known that they was liked by the users
    predicted_ratings: array of numpy arrays
        item ratings that were recommended to the users

    Returns
    -------
    Mean of MSE for all users: float
    """

    def one_mean_square_error(index) -> float:
        return _one_mean_square_error(true_ratings[index], predicted_ratings[index])

    if type(true_ratings) != list or type(predicted_ratings) != list:
        raise TypeError('Input values need to have list format')

    if len(true_ratings) != len(predicted_ratings):
        raise ValueError('Two arrays should have same shape')

    ratings = np.arange(len(true_ratings))
    return np.vectorize(one_mean_square_error)(ratings).mean()
