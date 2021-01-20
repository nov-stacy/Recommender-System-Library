import numpy as np
import typing as tp
import scipy.sparse as sparse
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from recommender_system.models.abstract_recommender_system import RecommenderSystem


class NearestNeigborsModel(RecommenderSystem):
    """
    Distance-based recommender systems method between users and recommending what neighbors like
    """

    def __init__(self, k_nearest_neigbors: int) -> None:
        """
        :param k_nearest_neigbors: the number of closest neighbors that must be taken into account when predicting an
        estimate for an item
        """
        self.__knn: NearestNeighbors = NearestNeighbors(n_neighbors=k_nearest_neigbors + 1)
        self.__data: sparse.coo_matrix = np.array([])
        self.__mean_items: np.ndarray = np.array([])
        self.__mean_users: np.ndarray = np.array([])

    def __calculate_correlation_coefficients(self, user_index: int, users_indexes: tp.List[int]) -> np.ndarray:
        """
        Method to calculate correlation coefficients between users
        :param user_index: current user
        :param users_indexes: users ratio with which to get
        :return: correlation coefficients
        """
        return np.vectorize(lambda index: pearsonr(self.__data.getrow(user_index).toarray()[0],
                                                   self.__data.getrow(index).toarray()[0])[0])(users_indexes)

    def train(self, data: sparse.coo_matrix) -> 'NearestNeigborsModel':
        self.__data = data
        self.__mean_items = self.__data.mean(axis=0).transpose()
        self.__mean_users = self.__data.mean(axis=1)
        self.__knn.fit(self.__data)
        return self

    def retrain(self, data: sparse.coo_matrix) -> 'NearestNeigborsModel':
        return self.train(data)

    def predict(self, user_index: int) -> np.ndarray:
        # find a list of k nearest neigbors of current user
        nearest_users = self.__knn.kneighbors(self.__data.getrow(user_index), return_distance=False)[0]
        nearest_users = nearest_users[nearest_users != user_index]

        # get correlation coefficient and change nan values
        correlation_coefficients = np.nan_to_num(self.__calculate_correlation_coefficients(user_index, nearest_users))

        # get mean ratings of items and mean ratings given by users
        mean_users = self.__mean_users[nearest_users]

        # get ratings given by nearest users to product data
        ratings_users = sparse.vstack([self.__data.getrow(index) for index in nearest_users])

        # calculate ratings
        numerator = np.sum(np.multiply(ratings_users - mean_users,
                                       correlation_coefficients.reshape((nearest_users.shape[0], 1))), axis=0)
        denominator = np.sum(np.abs(correlation_coefficients))

        return np.squeeze(np.asarray(self.__mean_items + numerator.transpose() / denominator))
