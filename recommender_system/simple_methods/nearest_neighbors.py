import numpy as np
import typing as tp
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from recommender_system.abstract import RecommenderSystem


class NearestNeigborsMethod(RecommenderSystem):

    def __init__(self, k_nearest_neigbors: int) -> None:
        self.__knn: NearestNeighbors = NearestNeighbors(n_neighbors=k_nearest_neigbors + 1)
        self.__data: np.array = np.array([])
        self.__clear_data: np.array = np.array([])
        self.__mean_items: np.array = np.array([])
        self.__mean_users: np.array = np.array([])

    def __calculate_correlation_coefficients(self, user_index: int, users_indexes: tp.List[int]) -> np.array:
        return np.vectorize(lambda index: pearsonr(self.__data[user_index], self.__data[index])[0])(users_indexes)

    def __calculate_ranks(self, user_index: int) -> tp.List[tp.Tuple[int, int]]:
        nearest_users = self.__knn.kneighbors(self.__data[user_index].reshape(1, -1), return_distance=False)[0]
        nearest_users = nearest_users[nearest_users != user_index]
        correlation_coefficients = np.nan_to_num(self.__calculate_correlation_coefficients(user_index, nearest_users))
        mean_users = self.__mean_users[nearest_users]
        unknown_ranks = np.argwhere(np.isnan(self.__clear_data[user_index]))
        ranks_users = self.__data[nearest_users, unknown_ranks]
        numerator = np.sum((ranks_users - mean_users) * correlation_coefficients, axis=1)
        denominator = np.sum(np.abs(correlation_coefficients))
        mean_items = self.__mean_items[unknown_ranks].transpose()[0]
        return list(zip(mean_items + numerator / denominator, unknown_ranks[:, 0]))

    def train(self, data: np.array) -> 'NearestNeigborsMethod':
        self.__clear_data = data
        self.__data = np.nan_to_num(self.__clear_data)
        self.__mean_items = self.__data.mean(axis=0).transpose()
        self.__mean_users = self.__data.mean(axis=1)
        self.__knn.fit(self.__data)
        return self

    def retrain(self, data: np.array) -> 'NearestNeigborsMethod':
        return self.train(data)

    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        ranked_list = self.__calculate_ranks(user_index)
        ranked_list.sort(reverse=True)
        return np.array(list(zip(*ranked_list[:k_items]))[1])
