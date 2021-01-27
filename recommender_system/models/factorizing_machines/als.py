import numpy as np
import random
from scipy import sparse as sparse

from recommender_system.models.abstract_recommender_system import RecommenderSystem


class AlternatingLeastSquaresModel(RecommenderSystem):
    """

    """

    def __init__(self, dimension: int, learning_rate: float, iteration_number: int = 1000,
                 user_regularization: float = 0, item_regularization: float = 0) -> None:
        """
        :param dimension: the number of singular values to keep
        :param learning_rate: learning rate for stochastic gradient descent
        :param iteration_number: number of iterations in stochastic gradient descent
        :param user_regularization: regularization member for user data
        :param item_regularization: regularization member for item data
        """
        self.__dimension: int = dimension
        self.__learning_rate: float = learning_rate
        self.__iteration_number: int = iteration_number
        self.__user_regularization: float = user_regularization
        self.__item_regularization: float = item_regularization
        self.__data: sparse.coo_matrix = np.array([])
        self.__user_matrix: np.ndarray = np.array([])
        self.__item_matrix: np.ndarray = np.array([])

    def __use_stochastic_gradient_descent(self) -> None:
        """

        """

        non_null_data_shape: int = self.__data.data.shape[0]

        for _ in range(self.__iteration_number):

            random_index: int = random.randint(0, self.__data.row.shape[0] - 1)
            random_user_index: int = self.__data.row[random_index]
            random_item_index: int = self.__data.col[random_index]
            rating: float = self.__data.data[random_index]

            delta_approximation = rating - self.__user_matrix[random_user_index] @ \
                                  self.__item_matrix[random_item_index].T

            self.__user_matrix[random_user_index] += self.__learning_rate * delta_approximation * \
                                              self.__item_matrix[random_item_index] - 2 * self.__user_regularization * \
                                              np.sum(self.__user_matrix[random_user_index]) / non_null_data_shape
            self.__item_matrix[random_item_index] += self.__learning_rate * delta_approximation * \
                                              self.__user_matrix[random_user_index] - 2 * self.__item_regularization * \
                                              np.sum(self.__item_matrix[random_item_index]) / non_null_data_shape

    def train(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        self.__data = data.copy()

        self.__user_matrix: np.ndarray = np.zeros(shape=(self.__data.shape[0], self.__dimension))
        self.__item_matrix: np.ndarray = np.zeros(shape=(self.__data.shape[1], self.__dimension))

        self.__use_stochastic_gradient_descent()

        return self

    def retrain(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        self.__use_stochastic_gradient_descent()
        return self

    def predict(self, user_index: int) -> np.ndarray:
        return self.__user_matrix[user_index] @ self.__item_matrix.T
