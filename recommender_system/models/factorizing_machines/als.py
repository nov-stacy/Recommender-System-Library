import numpy as np
import typing as tp
from scipy import sparse as sparse
from recommender_system.models.abstract_recommender_system import RecommenderSystem
from tqdm import tqdm


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

        self.__debug_information: tp.List[float] = []

    def __calculate_debug_metric(self) -> float:
        result: float = 0
        for user_index, item_index, data in zip(self.__data.row, self.__data.col, self.__data.data):
            result += (data - self.__user_matrix[user_index] @ self.__item_matrix[item_index].T) ** 2
        return result

    def __use_stochastic_gradient_descent(self, epochs: int, debug: bool) -> None:
        """
        Method for stochastic gradient descent

        Parameters
        ----------
        epochs: int
            The number of epochs that the method must pass
        debug: bool
            TODO
        """

        non_null_data_shape: int = self.__data.data.shape[0]  # amount of non-zero data
        learning_rate = self.__learning_rate
        self.__debug_information: tp.List[float] = []

        for epoch in tqdm(range(epochs)):

            shuffle_indices = np.arange(non_null_data_shape)
            np.random.shuffle(shuffle_indices)

            for index in shuffle_indices:

                # get indices for user and item and rating
                user_index: int = self.__data.row[index]
                item_index: int = self.__data.col[index]
                rating: float = self.__data.data[index]

                # get the difference between the true value of the rating and the approximate
                delta_approximation = rating - self.__user_matrix[user_index] @ \
                                      self.__item_matrix[item_index].T

                # change user weights
                user_regularization_add = 2 * self.__user_regularization * \
                                          np.sum(self.__user_matrix[user_index]) / non_null_data_shape
                self.__user_matrix[user_index] += learning_rate * (delta_approximation * self.__item_matrix[item_index]
                                                                   - user_regularization_add)

                # change item weights
                item_regularization_add = 2 * self.__item_regularization * \
                                          np.sum(self.__item_matrix[item_index]) / non_null_data_shape
                self.__item_matrix[item_index] += learning_rate * (delta_approximation * self.__user_matrix[user_index]
                                                                   - item_regularization_add)

            # learning_rate *= 1 / np.log(epoch)

            if debug:
                value = self.__calculate_debug_metric()
                self.__debug_information.append(value)

    def get_debug_information(self) -> tp.List[float]:
        return self.__debug_information

    def train(self, data: sparse.coo_matrix,
              epochs: int = 100,  # The number of epochs that the method must pass
              debug=False  # TODO
              ) -> 'RecommenderSystem':

        self.__data = data.copy()

        # generate matrices with latent features
        self.__user_matrix: np.ndarray = np.random.randn(self.__data.shape[0], self.__dimension)
        self.__item_matrix: np.ndarray = np.random.randn(self.__data.shape[1], self.__dimension)

        self.__use_stochastic_gradient_descent(epochs, debug)

        return self

    def retrain(self, data: sparse.coo_matrix,
                epochs: int = 100,  # The number of epochs that the method must pass
                debug: bool = False  # TODO
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
        self.__use_stochastic_gradient_descent(epochs, debug)

        return self

    def predict(self, user_index: int) -> np.ndarray:
        return self.__user_matrix[user_index] @ self.__item_matrix.T
