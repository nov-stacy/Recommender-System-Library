import numpy as np
import typing as tp
from recommender_system.abstract import RecommenderSystem


class SingularValueDecompositionNaiveModel(RecommenderSystem):
    """
    Recommender system based on singular value decomposition
    """

    def __init__(self, dimension: int) -> None:
        self.__dimension: int = dimension
        self.__first_matrix: np.array = np.array([])
        self.__second_matrix: np.array = np.array([])
        self.__singular_values: np.array = np.array([])
        self.__data: np.array = np.array([])

    def __calculate_ratings(self, user_index: int) -> tp.List[tp.Tuple[int, int]]:
        # find items that user didnt mark
        unknown_ratings = np.argwhere(np.isnan(self.__data[user_index]))[:, 0]

        # all ratings for current user
        all_ratings_for_user = self.__first_matrix[2] @ np.diag(self.__singular_values) @ self.__second_matrix

        # predict ratings for items that user didnt mark
        predict_ratings = all_ratings_for_user[unknown_ratings]

        return list(zip(predict_ratings, unknown_ratings))

    def train(self, data: np.array) -> 'SingularValueDecompositionNaiveModel':
        first_matrix, singular_values, second_matrix = np.linalg.svd(np.nan_to_num(data), full_matrices=False)
        self.__first_matrix = first_matrix[:, : self.__dimension]
        self.__second_matrix = second_matrix[: self.__dimension, :]
        self.__singular_values = singular_values[: self.__dimension]
        self.__data = data
        return self

    def retrain(self, data: np.array) -> 'SingularValueDecompositionNaiveModel':
        return self.train(data)

    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        ranked_list = self.__calculate_ratings(user_index)
        ranked_list.sort(reverse=True)
        return np.array(list(zip(*ranked_list[:k_items]))[1])

