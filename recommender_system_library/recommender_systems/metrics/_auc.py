import typing as tp
import numpy as np


__all__ = ['roc_auc']


def _one_roc_auc(true_interest: np.ndarray, predicted_ratings: np.ndarray) -> float:
    """
    Method to calculate ROC-AUC for one user

    Parameters
    ----------
    true_interest: numpy array
        Indicators of items, about which it is known that they was liked or unliked by the users
    predicted_ratings: numpy array
        Item ratings that were recommended to the user

    Returns
    -------
    ROC-AUC for user: float
    """

    from sklearn.metrics import roc_auc_score

    if type(true_interest) != np.ndarray or type(predicted_ratings) != np.ndarray:
        raise TypeError('Ratings need to have numpy array format')

    if true_interest.shape != predicted_ratings.shape or len(true_interest.shape) != 1:
        raise ValueError('True and predicted ratings need to be 1D array with same shape')

    if true_interest.shape[0] and true_interest.dtype != np.int or \
            predicted_ratings.shape[0] and predicted_ratings.dtype not in [np.float, np.int]:
        raise TypeError('True interest should store int values and ratings should be float or int')

    if set(true_interest) != {0, 1}:
        raise ValueError('True interest should store 0 and 1')

    return roc_auc_score(true_interest, predicted_ratings)


def roc_auc(true_interest: tp.List[np.ndarray], predicted_ratings: tp.List[np.ndarray]) -> float:
    """
    Method to calculate ROC-AUC for all users

    Parameters
    ----------
    true_interest: array of numpy arrays
        Indicators of items, about which it is known that they was liked or unliked by the users (for example: [0, 0, 1])
    predicted_ratings: array of numpy arrays
        Item ratings that were recommended to the users

    Returns
    -------
    Mean of ROC-AUC for all users: float
    """

    def one_roc_auc(index) -> float:
        return _one_roc_auc(true_interest[index], predicted_ratings[index])

    if type(true_interest) != list or type(predicted_ratings) != list:
        raise TypeError('Input values need to have list format')

    if len(true_interest) != len(predicted_ratings):
        raise ValueError('Two arrays should have same shape')

    ratings = np.arange(len(true_interest))
    return np.vectorize(one_roc_auc)(ratings).mean()
