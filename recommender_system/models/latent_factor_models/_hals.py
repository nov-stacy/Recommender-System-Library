import numpy as np
from scipy import sparse as sparse
from tqdm import tqdm

from recommender_system.functional_errors.latent_error import calculate_error_for_latent_models
from recommender_system.models.abstract import RecommenderSystem, DebugInterface


class HierarchicalAlternatingLeastSquaresModel(RecommenderSystem, DebugInterface):

    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super(DebugInterface, self).__init__(calculate_error_for_latent_models)

        self.__dimension: int = dimension

        self.__user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self.__item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.__users_count: int = 0  # number of users in the system
        self.__items_count: int = 0  # number of items in the system

    def __calculate_delta(self, data: sparse.coo_matrix, index: int) -> np.array:
        """
        Method for calculate the difference between the original rating matrix and the matrix
        that was obtained at this point in time
        """

        indices = list(range(index)) + list(range(index + 1, self.__dimension))
        return data - sum((self.__item_matrix[:, _index] @ self.__item_matrix[:, _index].T for _index in indices))

    def __calculate_user_matrix(self, index: int, delta: np.array) -> None:
        """
        Method for finding a column of users matrix

        Parameters
        ----------
        index: int
            Index of the hidden attribute
        delta: numpy array
            The difference between the original rating matrix and the matrix that was obtained at this point in time
        """

        denominator = self.__item_matrix[:, index].T @ self.__item_matrix[:, index]
        self.__user_matrix[:, index] = delta @ self.__item_matrix[:, index] / denominator

    def __calculate_item_matrix(self, index: int, delta: np.array) -> None:
        """
        Method for finding a column of items matrix

        Parameters
        ----------
        index: int
            Index of the hidden attribute
        delta: numpy array
            The difference between the original rating matrix and the matrix that was obtained at this point in time
        """

        denominator = self.__user_matrix[:, index].T @ self.__user_matrix[:, index]
        self.__user_matrix[:, index] = self.__user_matrix[:, index].T @ delta / denominator

    def __start_train(self, data: sparse.coo_matrix, epochs: int, is_debug: bool):
        """
        Method for training a recommendation system

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch
        """

        # determining average values for users and items
        mean_users: np.ndarray = data.mean(axis=1)
        mean_items: np.ndarray = data.mean(axis=0).transpose()

        # determining known ratings
        users_indices: np.ndarray = data.row
        items_indices: np.ndarray = data.col
        ratings: np.ndarray = data.data

        self.__update_debug_information(is_debug)

        for _ in tqdm(range(epochs)):
            for index in range(self.__dimension):

                # the difference between the original rating matrix and user_matrix @ item_matrix
                delta = self.__calculate_delta(data, index)

                # calculate  a columns of matrices for users and items
                self.__calculate_user_matrix(index, delta)
                self.__calculate_item_matrix(index, delta)

            self.__set_debug_information(is_debug, self.__user_matrix, self.__item_matrix, mean_users, mean_items,
                                         users_indices, items_indices, ratings)

    def train(self, data: sparse.coo_matrix,
              epochs: int = 100,  # The number of epochs that the method must pass
              is_debug: bool = False  # Indicator of the need to maintain the error functionality on each epoch
              ) -> 'RecommenderSystem':

        # determining the number of users and items
        self.__users_count: int = data.shape[0]
        self.__items_count: int = data.shape[1]

        # generate matrices with latent features
        self.__user_matrix: np.ndarray = np.random.randn(self.__users_count, self.__dimension)
        self.__item_matrix: np.ndarray = np.random.randn(self.__items_count, self.__dimension)

        self.__start_train(data, epochs, is_debug)

    def retrain(self, data: sparse.coo_matrix,
                epochs: int = 100,  # The number of epochs that the method must pass
                is_debug: bool = False  # Indicator of the need to maintain the error functionality on each epoch
                ) -> 'RecommenderSystem':

        # add new users to matrix
        if data.shape[0] > self.__users_count:
            matrix_part = np.random.randn(data.shape[0] - self.__users_count, self.__dimension)
            self.__user_matrix = np.vstack((self.__user_matrix, matrix_part))

        # add new items to matrix
        if data.shape[1] > self.__items_count:
            matrix_part = np.random.randn(data.shape[1] - self.__items_count, self.__dimension)
            self.__item_matrix = np.vstack((self.__item_matrix, matrix_part))

        self.__start_train(data, epochs, is_debug)

        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:
        return self.__user_matrix[user_index] @ self.__item_matrix.T
