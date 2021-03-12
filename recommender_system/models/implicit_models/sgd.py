import numpy as np
from scipy import sparse as sparse
from tqdm import tqdm

from recommender_system.models.abstract_recommender_system import RecommenderSystem


class StochasticImplicitLatentFactorModel(RecommenderSystem):
    """
    A model based on the fact that any signals from the user are better than their absence.

    Realization
    -----------

    """

    def __init__(self, dimension: int, learning_rate: float, influence_regularization: float = 0,
                 user_regularization: float = 0, item_regularization: float = 0) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        learning_rate: float
            Learning rate for stochastic gradient descent
        influence_regularization: float
            The regularization coefficient of the effect of an explicit rating on confidence in interest
        user_regularization: float
            Regularization member for user data
        item_regularization: float
            Regularization member for item data
        """

        self.__dimension: int = dimension
        self.__rate: float = learning_rate
        self.__influence: float = influence_regularization
        self.__user_regularization: float = user_regularization
        self.__item_regularization: float = item_regularization

        self.__user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self.__item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.__users_count: int = 0  # number of users in the system
        self.__items_count: int = 0  # number of items in the system

    def __use_stochastic_gradient_descent(self, data: sparse.coo_matrix, epochs: int,
                                          mean_ii_user: np.ndarray,
                                          mean_ii_item: np.ndarray) -> None:
        """
        Method for stochastic gradient descent

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        mean_ii_user: numpy array

        mean_ii_item: numpy array

        """

        for epoch in tqdm(range(epochs)):

            # random users
            users_indices = np.arange(self.__users_count)
            np.random.shuffle(users_indices)

            for user_index in users_indices:

                # random items
                items_indices = np.arange(self.__items_count)
                np.random.shuffle(items_indices)

                # ratings that the user has set for items
                user_ratings: np.ndarray = data.getrow(user_index).toarray()
                # indicators of implicit user interest in items
                ii_user: np.ndarray = (user_ratings != 0).astype(int)

                for item_index in items_indices:

                    # similarity between user and item
                    similarity = self.__user_matrix[user_index] @ self.__item_matrix[item_index].T

                    # get the difference between the true value of the rating and the approximate
                    delta = (ii_user[item_index] - mean_ii_user[user_index] -
                             mean_ii_item[item_index] - similarity) * (1 + self.__influence * user_ratings[item_index])

                    # the value of regularization for the user
                    user_reg = self.__user_regularization * np.sum(self.__user_matrix[user_index]) / self.__dimension
                    # changing hidden variables for the user
                    self.__user_matrix[user_index] += self.__rate * (delta * self.__item_matrix[item_index] - user_reg)

                    # the value of regularization for the item
                    item_reg = self.__item_regularization * np.sum(self.__item_matrix[item_index]) / self.__dimension
                    # changing hidden variables for the item
                    self.__item_matrix[item_index] += self.__rate * (delta * self.__user_matrix[user_index] - item_reg)

    def __start_train(self, data: sparse.coo_matrix, epochs: int):
        """
        Method for processing data before training

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        """

        # determining the number of users and items
        self.__users_count: int = data.shape[0]
        self.__items_count: int = data.shape[1]

        # determining the average values of implicit interest for users and items
        implicit_user_mean_interest = (data != 0).astype(int).mean(axis=1)
        implicit_item_mean_interest = (data != 0).mean(axis=0).transpose()

        # start train
        self.__use_stochastic_gradient_descent(data, epochs, implicit_user_mean_interest, implicit_item_mean_interest)

    def train(self, data: sparse.coo_matrix,
              epochs: int = 100,  # The number of epochs that the method must pass
              is_debug: bool = False  # Indicator of the need to maintain the error functionality on each epoch
              ) -> 'RecommenderSystem':

        # generate matrices with latent features
        self.__user_matrix: np.ndarray = np.random.randn(data.shape[0], self.__dimension)
        self.__item_matrix: np.ndarray = np.random.randn(data.shape[1], self.__dimension)

        self.__start_train(data, epochs)

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

        self.__start_train(data, epochs)

        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:
        return self.__user_matrix[user_index] @ self.__item_matrix.T
