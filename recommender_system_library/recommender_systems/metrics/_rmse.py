import typing as tp
import numpy as np

__all__ = ['root_mean_square_error']

from recommender_systems.metrics._mse import _one_mean_square_error


def root_mean_square_error(true_ratings: tp.List[np.ndarray], predicted_ratings: tp.List[np.ndarray]) -> float:
    """
    Method to calculate RMSE for all users

    Parameters
    ----------
    true_ratings: array of numpy arrays
        Ratings of items, about which it is known that they was liked by the users
    predicted_ratings: array of numpy arrays
        item ratings that were recommended to the users

    Returns
    -------
    Mean of RMSE for all users: float
    """

    def one_root_mean_square_error(index) -> float:
        return np.sqrt(_one_mean_square_error(true_ratings[index], predicted_ratings[index]))

    if type(true_ratings) != list or type(predicted_ratings) != list:
        raise TypeError('Input values need to have list format')

    if len(true_ratings) != len(predicted_ratings):
        raise ValueError('Two arrays should have same shape')

    ratings = np.arange(len(true_ratings))
    return np.vectorize(one_root_mean_square_error)(ratings).mean()
