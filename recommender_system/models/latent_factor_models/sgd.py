import typing as tp

import numpy as np
from scipy import sparse
from tqdm import tqdm

from recommender_system.models.abstract_recommender_system import RecommenderSystem


class StochasticLatentFactorModel(RecommenderSystem):
    """
    A model with hidden variables.

    Realization
    -----------
    Vectors denoting categories of interests are constructed for each user and item.
    Such vectors are representations that allow you to reduce entities into a single vector space.
    The model is trained using stochastic gradient descent, which randomly shuffles all known ratings at each epoch
    and goes through them.
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
        self.__rate: float = learning_rate
        self.__user_regularization: float = user_regularization
        self.__item_regularization: float = item_regularization

        self.__user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self.__item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.__users_count: int = 0  # number of users in the system
        self.__items_count: int = 0  # number of items in the system

        self.__debug_information: tp.Optional[tp.List[float]] = None  # array with errors on each epoch

    def __calculate_error(self, mean_users: np.ndarray, mean_items: np.ndarray, users_indices: np.ndarray,
                          items_indices: np.ndarray, ratings: np.ndarray,) -> float:
        """
        Method for determining the error functional in a model with hidden variables

        Parameters
        ----------
        mean_users: numpy array
            Array with the average for each user
        mean_items: numpy array
            Array with the average for each item
        users_indices: numpy array
            Indices of users for whom ratings are known
        items_indices: numpy array
            Indices of items for whom ratings are known
        ratings: numpy array
            Known ratings
        """

        result: float = 0  # error functionality

        # for each user and item for which the ratings are known
        for user_index, item_index, rating in zip(users_indices, items_indices, ratings):

            # similarity between user and item
            similarity = self.__user_matrix[user_index] @ self.__item_matrix[item_index].T
            # adding to the functionality
            result += (rating - mean_users[user_index] - mean_items[item_index] - similarity) ** 2

        return result

    def __use_stochastic_gradient_descent(self, epochs: int, mean_users: np.ndarray, mean_items: np.ndarray,
                                          users_indices: np.ndarray, items_indices: np.ndarray,
                                          ratings: np.ndarray, is_debug: bool = False) -> None:
        """
        Method for stochastic gradient descent

        Parameters
        ----------
        epochs: int
            The number of epochs that the method must pass
        mean_users: numpy array
            Array with the average for each user
        mean_items: numpy array
            Array with the average for each item
        users_indices: numpy array
            Indices of users for whom ratings are known
        items_indices: numpy array
            Indices of items for whom ratings are known
        ratings: numpy array
            Known ratings
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch
        """

        self.__debug_information = [] if is_debug else None

        for _ in tqdm(range(epochs)):

            shuffle_indices = np.arange(ratings.shape[0])
            np.random.shuffle(shuffle_indices)

            for index in shuffle_indices:

                # get indices for user and item and rating
                user_index: int = users_indices[index]
                item_index: int = items_indices[index]
                rating: float = ratings[index]

                # similarity between user and item
                similarity = self.__user_matrix[user_index] @ self.__item_matrix[item_index].T
                # get the difference between the true value of the rating and the approximate
                delta = rating - mean_users[user_index] - mean_items[item_index] - similarity

                # the value of regularization for the user
                user_reg = self.__user_regularization * np.sum(self.__user_matrix[user_index]) / self.__dimension
                # changing hidden variables for the user
                self.__user_matrix[user_index] += self.__rate * (delta * self.__item_matrix[item_index] - user_reg)

                # the value of regularization for the item
                item_reg = self.__item_regularization * np.sum(self.__item_matrix[item_index]) / self.__dimension
                # changing hidden variables for the item
                self.__item_matrix[item_index] += self.__rate * (delta * self.__user_matrix[user_index] - item_reg)

            if is_debug:
                # calculate error functionality
                error = self.__calculate_error(mean_users, mean_items, users_indices, items_indices, ratings)
                self.__debug_information.append(error)

    def __start_train(self, data: sparse.coo_matrix, epochs: int, is_debug: bool):
        """
        Method for processing data before training

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

        # determining the number of users and items
        self.__users_count: int = data.shape[0]
        self.__items_count: int = data.shape[1]

        # determining average values for users and items
        mean_users: np.ndarray = data.mean(axis=1)
        mean_items: np.ndarray = data.mean(axis=0).transpose()

        # determining known ratings
        users_indices: np.ndarray = data.row
        items_indices: np.ndarray = data.col
        ratings: np.ndarray = data.data

        # start train
        self.__use_stochastic_gradient_descent(epochs, mean_users, mean_items,
                                               users_indices, items_indices, ratings, is_debug)

    def train(self, data: sparse.coo_matrix,
              epochs: int = 100,  # The number of epochs that the method must pass
              is_debug: bool = False  # Indicator of the need to maintain the error functionality on each epoch
              ) -> 'RecommenderSystem':

        # generate matrices with latent features
        self.__user_matrix: np.ndarray = np.random.randn(data.shape[0], self.__dimension)
        self.__item_matrix: np.ndarray = np.random.randn(data.shape[1], self.__dimension)

        self.__start_train(data, epochs, is_debug)

        return self

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

    def get_debug_information(self) -> tp.List[float]:
        """
        Method for getting a list of functionality errors that were optimized during training

        Returns
        -------
        list of functionality errors: list[float]
        """

        if self.__debug_information is None:
            raise AttributeError("No debug information because there was is_debug = False during training")
        else:
            return self.__debug_information
