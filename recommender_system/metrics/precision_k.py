import typing as tp
import numpy as np


def precision_for_one_user(true_user_pref_indices: np.ndarray,
                           predicted_user_pref_indices: np.ndarray) -> float:
    """
    Method to calculate Precision@k for one user
    (Recall at k is the proportion of relevant items found in the top-k recommendations)
    :param true_user_pref_indices: indices of items, about which it is known that they was liked by the user
    :param predicted_user_pref_indices: item indices that were recommended to the user
    :return: Precision@k for user
    """
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
    :param true_preferences_indices: indices of items, about which it is known that they was liked by the users
    :param predicted_preferences_indices: item indices that were recommended to the users
    :return: mean of Precision@k for all users
    """

    if len(true_preferences_indices) != len(predicted_preferences_indices):
        raise ValueError('Two arrays should have same shape')

    indices = np.arange(len(true_preferences_indices))
    return np.vectorize(lambda index: precision_for_one_user(true_preferences_indices[index],
                                                             predicted_preferences_indices[index]))(indices).mean()
