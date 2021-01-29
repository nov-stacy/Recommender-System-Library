import numpy as np
import random
from scipy import sparse as sparse
from recommender_system.models.abstract_recommender_system import RecommenderSystem


class AlternatingLeastSquaresModel(RecommenderSystem):
    """
    Recommender system based on stochastic gradient descent and regularization

    Realization
    -----------
    Optimization problem
    sum((p_{ij} - (b_i, c_j))^2) + a * sum(||b_i||^2) + b * sum(||c_j||^2) -> min
    for p_{ij} != None
    """

    def __init__(self, dimension: int, learning_rate: float,
                 user_regularization: float = 0, item_regularization: float = 0) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        learning_rate: float
            Learning rate for stochastic gradient descent
        user_regularization: float
            Regularization member for user data
        item_regularization: float
            Regularization member for item data
        """
        self.__dimension: int = dimension
        self.__learning_rate: float = learning_rate
        self.__user_regularization: float = user_regularization
        self.__item_regularization: float = item_regularization

        self.__data: sparse.coo_matrix = np.array([])  # saved data
        self.__user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self.__item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

    def __use_stochastic_gradient_descent(self, iteration_number: int) -> None:
        """
        Method for stochastic gradient descent

        Parameters
        ----------
        iteration_number: int
            The number of iterations that the method must pass
        """

        non_null_data_shape: int = self.__data.data.shape[0]  # amount of non-zero data

        for _ in range(iteration_number):

            # get random index: random user and random item
            random_index: int = random.randint(0, self.__data.row.shape[0] - 1)
            random_user_index: int = self.__data.row[random_index]
            random_item_index: int = self.__data.col[random_index]
            rating: float = self.__data.data[random_index]

            # get the difference between the true value of the rating and the approximate
            delta_approximation = rating - self.__user_matrix[random_user_index] @ \
                                  self.__item_matrix[random_item_index].T

            # change user weights
            self.__user_matrix[random_user_index] += self.__learning_rate * (
                    delta_approximation * self.__item_matrix[random_item_index] - 2 * self.__user_regularization *
                    np.sum(self.__user_matrix[random_user_index]) / non_null_data_shape)

            # change item weights
            self.__item_matrix[random_item_index] += self.__learning_rate * (
                    delta_approximation * self.__user_matrix[random_user_index] - 2 * self.__item_regularization *
                    np.sum(self.__item_matrix[random_item_index]) / non_null_data_shape)

    def train(self, data: sparse.coo_matrix,
              iteration_number: int = 1000  # The number of iterations that the method must pass
              ) -> 'RecommenderSystem':

        self.__data = data.copy()

        # generate matrices with latent features
        self.__user_matrix: np.ndarray = np.random.randn(self.__data.shape[0], self.__dimension)
        self.__item_matrix: np.ndarray = np.random.randn(self.__data.shape[1], self.__dimension)

        self.__use_stochastic_gradient_descent(iteration_number)

        return self

    def retrain(self, data: sparse.coo_matrix,
                iteration_number: int = 1000  # The number of iterations that the method must pass
                ) -> 'RecommenderSystem':

        # add new users to matrix
        if data.shape[0] > self.__data.shape[0]:
            self.__user_matrix = np.vstack((self.__user_matrix,
                                            np.random.randn(data.shape[0] - self.__data.shape[0],
                                                                                self.__dimension)))

        # add new items to matrix
        if data.shape[1] > self.__data.shape[1]:
            self.__item_matrix = np.vstack((self.__item_matrix,
                                            np.random.randn(data.shape[0] - self.__data.shape[0],
                                                            self.__dimension)))

        self.__data = data.copy()
        self.__use_stochastic_gradient_descent(iteration_number)

        return self

    def predict(self, user_index: int) -> np.ndarray:
        return self.__user_matrix[user_index] @ self.__item_matrix.T
