import numpy as np
import typing as tp
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from recommender_system.abstract import RecommenderSystem


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
        self.__clear_data: np.array = np.array([])
        self.__data: np.array = np.array([])
        self.__mean_items: np.array = np.array([])
        self.__mean_users: np.array = np.array([])

    def __calculate_correlation_coefficients(self, user_index: int, users_indexes: tp.List[int]) -> np.array:
        """
        Method to calculate correlation coefficients between users
        :param user_index: current user
        :param users_indexes: users ratio with which to get
        :return: correlation coefficients
        """
        return np.vectorize(lambda index: pearsonr(self.__clear_data[user_index], self.__clear_data[index])[0])(users_indexes)

    def __calculate_ratings(self, user_index: int) -> tp.List[tp.Tuple[int, int]]:
        """
        Method to calculate ratings to items that user didnt mark
        :param user_index: the index of the user to make the prediction
        :return: list of elements of (rating, index_item) for each item
        """
        # find a list of k nearest neigbors of current user
        nearest_users = self.__knn.kneighbors(self.__clear_data[user_index].reshape(1, -1), return_distance=False)[0]
        nearest_users = nearest_users[nearest_users != user_index]

        # get correlation coefficient and change nan values
        correlation_coefficients = np.nan_to_num(self.__calculate_correlation_coefficients(user_index, nearest_users))

        # find items that user didnt mark
        unknown_ratings = np.argwhere(np.isnan(self.__data[user_index]))

        # get ratings given by nearest users to product data
        ratings_users = self.__clear_data[nearest_users, unknown_ratings]

        # get mean ratings of items and mean ratings given by users
        mean_users = self.__mean_users[nearest_users]
        mean_items = self.__mean_items[unknown_ratings].transpose()[0]

        # calculate ratings
        numerator = np.sum((ratings_users - mean_users) * correlation_coefficients, axis=1)
        denominator = np.sum(np.abs(correlation_coefficients))
        return list(zip(mean_items + numerator / denominator, unknown_ratings[:, 0]))

    def train(self, data: np.array) -> 'NearestNeigborsModel':
        self.__data = data
        self.__clear_data = np.nan_to_num(self.__data)
        self.__mean_items = self.__clear_data.mean(axis=0).transpose()
        self.__mean_users = self.__clear_data.mean(axis=1)
        self.__knn.fit(self.__clear_data)
        return self

    def retrain(self, data: np.array) -> 'NearestNeigborsModel':
        return self.train(data)

    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        ranked_list = self.__calculate_ratings(user_index)
        ranked_list.sort(reverse=True)
        return np.array(list(zip(*ranked_list[:k_items]))[1])
