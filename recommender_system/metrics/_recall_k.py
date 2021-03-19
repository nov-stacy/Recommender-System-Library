import typing as tp
import numpy as np


def __recall_for_one_user(true_user_pref_indices: np.ndarray,
                          predicted_user_pref_indices: np.ndarray) -> float:
    """
    Method to calculate Recall@k for one user

    Parameters
    ----------
    true_user_pref_indices: numpy array
        Indices of items, about which it is known that they was liked by the user
    predicted_user_pref_indices: numpy array
        Item indices that were recommended to the user

    Returns
    -------
    Recall@k for user: float
    """
    if true_user_pref_indices.dtype != np.int or predicted_user_pref_indices.dtype != np.int:
        raise ValueError('Arrays should store indices')

    if np.count_nonzero(true_user_pref_indices < 0) != 0 or np.count_nonzero(predicted_user_pref_indices < 0) != 0:
        raise ValueError('Arrays should store non-negative indices')

    if true_user_pref_indices.shape[0] == 0:
        return 0

    true_predicted_pref_number: float = len(set(predicted_user_pref_indices).intersection(true_user_pref_indices))
    return true_predicted_pref_number / true_user_pref_indices.shape[0]


def recall_k(true_preferences_indices: tp.List[np.ndarray],
             predicted_preferences_indices: tp.List[np.ndarray]) -> float:
    """
    Method to calculate Recall@k for all users
    (Recall at k is the proportion of relevant items found in the top-k recommendations)

    Parameters
    ----------
    true_preferences_indices: array of numpy arrays
        Indices of items, about which it is known that they was liked by the users
    predicted_preferences_indices: array of numpy arrays
        Item indices that were recommended to the users

    Returns
    -------
    Mean of Recall@k for all users: float
    """

    if len(true_preferences_indices) != len(predicted_preferences_indices):
        raise ValueError('Two arrays should have same shape')

    indices = np.arange(len(true_preferences_indices))
    return np.vectorize(lambda index: __recall_for_one_user(true_preferences_indices[index],
                                                            predicted_preferences_indices[index]))(indices).mean()