import typing as tp
import numpy as np


__all__ = ['recall_k']


def _one_user_recall(true_indices: np.ndarray, predicted_indices: np.ndarray) -> float:
    """
    Method to calculate Recall@k for one user

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
        If arrays don't store non-negative values

    Returns
    -------
    Recall@k for user: float
    """

    if type(true_indices) != np.ndarray or type(predicted_indices) != np.ndarray:
        raise TypeError('Indices need to have numpy array format')

    if true_indices.shape[0] and true_indices.dtype != np.int or \
            predicted_indices.shape[0] and predicted_indices.dtype != np.int:
        raise TypeError('Arrays should store indices')

    if np.count_nonzero(true_indices < 0) != 0 or np.count_nonzero(predicted_indices < 0) != 0:
        raise ValueError('Arrays should store non-negative indices')

    if true_indices.shape[0] == 0:
        return 0

    true_predicted_pref_number: float = len(set(predicted_indices).intersection(true_indices))
    return true_predicted_pref_number / true_indices.shape[0]


def recall_k(true_indices: tp.List[np.ndarray], predicted_indices: tp.List[np.ndarray]) -> float:
    """
    Method to calculate Recall@k for all users
    (Recall at k is the proportion of relevant items found in the top-k recommendations)

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
    Mean of Recall@k for all users: float
    """

    def one_user_recall(index) -> float:
        return _one_user_recall(true_indices[index], predicted_indices[index])

    if type(true_indices) != list or type(predicted_indices) != list:
        raise TypeError('Input values need to have list format')

    if len(true_indices) != len(predicted_indices):
        raise ValueError('Two arrays should have same shape')

    indices = np.arange(len(true_indices))
    return np.vectorize(one_user_recall)(indices).mean()
