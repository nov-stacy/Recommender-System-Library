import numpy as np


__all__ = ['calculate_error_for_implicit_models', 'calculate_error_for_latent_models']


def calculate_error_for_implicit_models():
    pass


def calculate_error_for_latent_models(user_matrix: np.array, item_matrix: np.array,
                                      mean_users: np.ndarray, mean_items: np.ndarray,
                                      users_indices: np.ndarray, items_indices: np.ndarray,
                                      ratings: np.ndarray) -> float:
    """
    Method for determining the error functional in a model with hidden variables

    Parameters
    ----------
    user_matrix: numpy array

    item_matrix: numpy array

    mean_users: numpy array
        Array with the average for each user
    mean_items: numpy array
        Array with the average for each item
    users_indices: numpy array
        Indices of users for whom ratings are known
    items_indices: numpy array
        Indices of items for whom ratings are known
    ratings: numpy array
        Known ratings
    """

    result: float = 0  # error functionality

    # for each user and item for which the ratings are known
    for user_index, item_index, rating in zip(users_indices, items_indices, ratings):
        # similarity between user and item
        similarity = user_matrix[user_index] @ item_matrix[item_index].T
        # adding to the functionality
        result += (rating - mean_users[user_index] - mean_items[item_index] - similarity) ** 2

    return result

