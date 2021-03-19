import numpy as np
from scipy import sparse as sparse
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from recommender_system.models.abstract import OneEpochAbstractRecommenderSystem


class UserBasedModel(OneEpochAbstractRecommenderSystem):
    """
    Recommender system based on distance between users and what neighbors like

    Realization
    -----------
    Determine the nearest neighbors for the user and determine ratings
    based on what they like and the distance to them
    """

    def __init__(self, k_nearest_neigbors: int) -> None:
        """
        Parameters
        ----------
        k_nearest_neigbors:
            The number of closest neighbors that must be taken into account
            when predicting an estimate for an item
        """
        self._data: sparse.coo_matrix = np.array([])  # matrix for storing all data
        self._mean_items: np.ndarray = np.array([])  # matrix for the average ratings of each user
        self._mean_users: np.ndarray = np.array([])  # matrix for the average ratings of each item

        # algorithm for determining the nearest neighbors
        self._k_nearest_neigbors = k_nearest_neigbors
        self._knn: NearestNeighbors = NearestNeighbors(n_neighbors=k_nearest_neigbors + 1)

    def train(self, data: sparse.coo_matrix) -> 'OneEpochAbstractRecommenderSystem':
        self._data: sparse.coo_matrix = data
        self._mean_items: np.ndarray = self._data.mean(axis=0).transpose()
        self._mean_users: np.ndarray = self._data.mean(axis=1)
        self._knn.fit(self._data)
        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:

        # get a list of k nearest neigbors of current user
        nearest_users: np.ndarray = self._knn.kneighbors(self._data.getrow(user_index), return_distance=False)[0]
        nearest_users = nearest_users[nearest_users != user_index]

        # get correlation coefficient and change nan values
        coeffs: np.ndarray = np.nan_to_num(
            np.vectorize(
                lambda index: pearsonr(self._data.getrow(user_index).toarray()[0],
                                       self._data.getrow(index).toarray()[0])[0]
            )(nearest_users)
        )

        # get mean ratings of items and mean ratings given by users
        mean_users: np.ndarray = self._mean_users[nearest_users]

        # get ratings given by nearest users to product data
        ratings_users: sparse.coo_matrix = sparse.vstack([self._data.getrow(index) for index in nearest_users])

        # calculate ratings
        numerator = np.sum(np.multiply(ratings_users - mean_users, coeffs.reshape((nearest_users.shape[0], 1))), axis=0)
        denominator = np.sum(np.abs(coeffs))

        return np.squeeze(np.asarray(self._mean_items + numerator.transpose() / denominator))

    def __str__(self) -> str:
        return f'User based [k_nearest_neigbors = {self._k_nearest_neigbors}]'
