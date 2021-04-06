import typing as tp
import numpy as np


def __precision_for_one_user(true_user_pref_indices: np.ndarray,
                             predicted_user_pref_indices: np.ndarray) -> float:
    """
    Method to calculate Precision@k for one user

    Parameters
    ----------
    true_user_pref_indices: numpy array
        Indices of items, about which it is known that they was liked by the user
    predicted_user_pref_indices: numpy array
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

    if type(true_user_pref_indices) != np.ndarray or type(predicted_user_pref_indices) != np.ndarray:
        raise TypeError('Indices need to have numpy array format')

    if true_user_pref_indices.dtype != np.int or predicted_user_pref_indices.dtype != np.int:
        raise ValueError('Arrays should store indices')

    if np.count_nonzero(true_user_pref_indices < 0) != 0 or np.count_nonzero(predicted_user_pref_indices < 0) != 0:
        raise ValueError('Arrays should store non-negative indices')

    if predicted_user_pref_indices.shape[0] == 0:
        return 0

    true_predicted_pref_number: float = len(set(predicted_user_pref_indices).intersection(true_user_pref_indices))
    return true_predicted_pref_number / predicted_user_pref_indices.shape[0]


def precision_k(true_preferences_indices: tp.List[np.ndarray],
                predicted_preferences_indices: tp.List[np.ndarray]) -> float:
    """
    Method to calculate Precision@k for all users
    (Precision at k is the proportion of recommended items in the top-k set that are relevant)

    Parameters
    ----------
    true_preferences_indices: array of numpy arrays
        Indices of items, about which it is known that they was liked by the users
    predicted_preferences_indices: array of numpy arrays
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

    if type(true_preferences_indices) != list or type(predicted_preferences_indices) != list:
        raise TypeError('Input values need to have list format')

    if len(true_preferences_indices) != len(predicted_preferences_indices):
        raise ValueError('Two arrays should have same shape')

    indices = np.arange(len(true_preferences_indices))
    return np.vectorize(lambda index: __precision_for_one_user(true_preferences_indices[index],
                                                               predicted_preferences_indices[index]))(indices).mean()
