import numpy as np
from scipy import sparse
from tqdm import tqdm

from recommender_system.functional_errors.latent_error import calculate_error_for_latent_models
from recommender_system.models.abstract import RecommenderSystem, DebugInterface


class StochasticLatentFactorModel(RecommenderSystem, DebugInterface):
    """
    A model with hidden variables.

    Realization
    -----------
    Vectors denoting categories of interests are constructed for each user and item.
    Such vectors are representations that allow you to reduce entities into a single vector space.
    The model is trained using stochastic gradient descent, which randomly shuffles all known ratings at each epoch
    and goes through them.
    """

    def __init__(self, dimension: int, learning_rate: float, user_regularization: float = 0,
                 item_regularization: float = 0) -> None:
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

        super(DebugInterface, self).__init__(calculate_error_for_latent_models)

        self.__dimension: int = dimension
        self.__rate: float = learning_rate
        self.__user_regularization: float = user_regularization
        self.__item_regularization: float = item_regularization

        self.__user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self.__item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.__users_count: int = 0  # number of users in the system
        self.__items_count: int = 0  # number of items in the system

    def __calculate_user_matrix(self, user_index: int, item_index: int, rating: float,
                                mean_users: np.ndarray, mean_items: np.ndarray,) -> None:
        # similarity between user and item
        similarity = self.__user_matrix[user_index] @ self.__item_matrix[item_index].T
        # get the difference between the true value of the rating and the approximate
        delta = rating - mean_users[user_index] - mean_items[item_index] - similarity

        # the value of regularization for the user
        user_reg = self.__user_regularization * np.sum(self.__user_matrix[user_index]) / self.__dimension
        # changing hidden variables for the user
        self.__user_matrix[user_index] += self.__rate * (delta * self.__item_matrix[item_index] - user_reg)

    def __calculate_item_matrix(self, user_index: int, item_index: int, rating: float,
                                mean_users: np.ndarray, mean_items: np.ndarray,) -> None:
        # similarity between user and item
        similarity = self.__user_matrix[user_index] @ self.__item_matrix[item_index].T
        # get the difference between the true value of the rating and the approximate
        delta = rating - mean_users[user_index] - mean_items[item_index] - similarity

        # the value of regularization for the item
        item_reg = self.__item_regularization * np.sum(self.__item_matrix[item_index]) / self.__dimension
        # changing hidden variables for the item
        self.__item_matrix[item_index] += self.__rate * (delta * self.__user_matrix[user_index] - item_reg)

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

        self.__update_debug_information(is_debug)

        for _ in tqdm(range(epochs)):

            shuffle_indices = np.arange(ratings.shape[0])
            np.random.shuffle(shuffle_indices)

            for index in shuffle_indices:

                # get indices for user and item and rating
                user_index: int = users_indices[index]
                item_index: int = items_indices[index]
                rating: float = ratings[index]

                self.__calculate_user_matrix(user_index, item_index, rating, mean_users, mean_items)
                self.__calculate_item_matrix(user_index, item_index, rating, mean_users, mean_items)

            self.__set_debug_information(is_debug, self.__user_matrix, self.__item_matrix, mean_users, mean_items,
                                         users_indices, items_indices, ratings)

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
