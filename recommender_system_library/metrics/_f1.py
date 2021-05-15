import typing as tp
import numpy as np


__all__ = ['f1_measure']

from recommender_system_library.metrics._precision_k import _one_user_precision
from recommender_system_library.metrics._recall_k import _one_user_recall


def f1_measure(true_indices: tp.List[np.ndarray], predicted_indices: tp.List[np.ndarray]) -> float:
    """
    Method to calculate F1 (symbiosis of precision and recall) of for all users

    Parameters
    ----------
    true_indices: array of numpy arrays
        Indices of items, about which it is known that they was liked by the users
    predicted_indices: array of numpy arrays
        Item indices that were recommended to the users

    Returns
    -------
    Mean of F1 for all users: float
    """

    def one_user_recall(index) -> float:
        precision = _one_user_precision(true_indices[index], predicted_indices[index])
        recall = _one_user_recall(true_indices[index], predicted_indices[index])
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    if type(true_indices) != list or type(predicted_indices) != list:
        raise TypeError('Input values need to have list format')

    if len(true_indices) != len(predicted_indices):
        raise ValueError('Two arrays should have same shape')

    indices = np.arange(len(true_indices))
    return np.vectorize(one_user_recall)(indices).mean()
