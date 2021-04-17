import typing as tp
import numpy as np


__all__ = ['precision_k']


def _one_user_precision(true_indices: np.ndarray, predicted_indices: np.ndarray) -> float:
    """
    Method to calculate Precision@k for one user

    Parameters
    ----------
    true_indices: numpy array
        Indices of items, about which it is known that they was liked by the user
    predicted_indices: numpy array
        Item indices that were recommended to the user

    Raises
    ------
    TypeError
        If parameters don't have needed format
    ValueError
        If arrays don't store int non-negative values

    Returns
    -------
    Precision@k for user: float
    """

    if type(true_indices) != np.ndarray or type(predicted_indices) != np.ndarray:
        raise TypeError('Indices need to have numpy array format')

    if true_indices.shape[0] and true_indices.dtype != np.int or \
            predicted_indices.shape[0] and predicted_indices.dtype != np.int:
        raise TypeError('Arrays should store indices')

    if np.count_nonzero(true_indices < 0) != 0 or np.count_nonzero(predicted_indices < 0) != 0:
        raise ValueError('Arrays should store non-negative indices')

    if predicted_indices.shape[0] == 0:
        return 0

    true_predicted_pref_number: float = len(set(predicted_indices).intersection(true_indices))
    return true_predicted_pref_number / predicted_indices.shape[0]


def precision_k(true_indices: tp.List[np.ndarray], predicted_indices: tp.List[np.ndarray]) -> float:
    """
    Method to calculate Precision@k for all users
    (Precision at k is the proportion of recommended items in the top-k set that are relevant)

    Parameters
    ----------
    true_indices: array of numpy arrays
        Indices of items, about which it is known that they was liked by the users
    predicted_indices: array of numpy arrays
        Item indices that were recommended to the users

    Raises
    ------
    TypeError
        If parameters don't have needed format
    ValueError
        If two arrays don't have same shape

    Returns
    -------
    Mean of Precision@k for all users: float
    """

    def one_user_precision(index) -> float:
        return _one_user_precision(true_indices[index], predicted_indices[index])

    if type(true_indices) != list or type(predicted_indices) != list:
        raise TypeError('Input values need to have list format')

    if len(true_indices) != len(predicted_indices):
        raise ValueError('Two arrays should have same shape')

    indices = np.arange(len(true_indices))
    return np.vectorize(one_user_precision)(indices).mean()
