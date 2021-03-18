import numpy as np
from scipy import sparse as sparse
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from recommender_system.models.abstract import RecommenderSystem


class UserBasedCollaborativeFilteringModel(RecommenderSystem):
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
        self.__data: sparse.coo_matrix = np.array([])  # matrix for storing all data
        self.__mean_items: np.ndarray = np.array([])  # matrix for the average ratings of each user
        self.__mean_users: np.ndarray = np.array([])  # matrix for the average ratings of each item

        # algorithm for determining the nearest neighbors
        self.__knn: NearestNeighbors = NearestNeighbors(n_neighbors=k_nearest_neigbors + 1)

    def train(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        self.__data: sparse.coo_matrix = data
        self.__mean_items: np.ndarray = self.__data.mean(axis=0).transpose()
        self.__mean_users: np.ndarray = self.__data.mean(axis=1)
        self.__knn.fit(self.__data)
        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:
        # get a list of k nearest neigbors of current user
        nearest_users: np.ndarray = self.__knn.kneighbors(self.__data.getrow(user_index), return_distance=False)[0]
        nearest_users = nearest_users[nearest_users != user_index]

        # get correlation coefficient and change nan values
        coeffs: np.ndarray = np.nan_to_num(
            np.vectorize(
                lambda index: pearsonr(self.__data.getrow(user_index).toarray()[0],
                                       self.__data.getrow(index).toarray()[0])[0]
            )(nearest_users)
        )

        # get mean ratings of items and mean ratings given by users
        mean_users: np.ndarray = self.__mean_users[nearest_users]

        # get ratings given by nearest users to product data
        ratings_users: sparse.coo_matrix = sparse.vstack([self.__data.getrow(index) for index in nearest_users])

        # calculate ratings
        numerator = np.sum(np.multiply(ratings_users - mean_users, coeffs.reshape((nearest_users.shape[0], 1))), axis=0)
        denominator = np.sum(np.abs(coeffs))

        return np.squeeze(np.asarray(self.__mean_items + numerator.transpose() / denominator))
