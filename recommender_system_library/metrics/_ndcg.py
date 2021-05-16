import typing as tp
import numpy as np

__all__ = ['normalized_discounted_cumulative_gain']


def _normalized_discounted_cumulative_gain(true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
    """
    Method to calculate NDCG for one user

    Parameters
    ----------
    true_ratings: numpy array
        Ratings of items, about which it is known that they was liked by the user
    predicted_ratings: numpy array
        Item ratings that were recommended to the user

    Returns
    -------
    NDCG for user: float
    """

    if type(true_ratings) != np.ndarray or type(predicted_ratings) != np.ndarray:
        raise TypeError('Ratings need to have numpy array format')

    if true_ratings.shape != predicted_ratings.shape or len(true_ratings.shape) != 1:
        raise ValueError('True and predicted ratings need to be 1D array with same shape')

    if true_ratings.shape[0] and true_ratings.dtype != np.float or \
            predicted_ratings.shape[0] and predicted_ratings.dtype != np.float:
        raise TypeError('Arrays should store ratings')

    sorted_ratings = np.sort(true_ratings)[::-1]
    sorted_ratings_by_pred = true_ratings[predicted_ratings.argsort()[::-1]]

    dcg = np.sum((2 ** sorted_ratings_by_pred - 1) / np.log2(1 + np.arange(1, sorted_ratings_by_pred.shape[0] + 1)))
    idcg = np.sum((2 ** sorted_ratings - 1) / np.log2(1 + np.arange(1, sorted_ratings.shape[0] + 1)))

    return dcg / idcg


def normalized_discounted_cumulative_gain(true_ratings: tp.List[np.ndarray], predicted_ratings: tp.List[np.ndarray]) -> float:
    """
    Method to calculate NDCG for all users

    Parameters
    ----------
    true_ratings: array of numpy arrays
        Ratings of items, about which it is known that they was liked by the users
    predicted_ratings: array of numpy arrays
        item ratings that were recommended to the users

    Returns
    -------
    Mean of NDCG for all users: float
    """

    def one_normalized_discounted_cumulative_gain(index) -> float:
        return _normalized_discounted_cumulative_gain(true_ratings[index], predicted_ratings[index])

    if type(true_ratings) != list or type(predicted_ratings) != list:
        raise TypeError('Input values need to have list format')

    if len(true_ratings) != len(predicted_ratings):
        raise ValueError('Two arrays should have same shape')

    ratings = np.arange(len(true_ratings))
    return np.vectorize(one_normalized_discounted_cumulative_gain)(ratings).mean()
