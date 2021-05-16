import numpy as np
from scipy import sparse

from recommender_system_library.models.abstract import EmbeddingsARS


class StochasticLatentFactorModel(EmbeddingsARS):
    """
    A model based only on the ratings.

    Realization
    -----------
    The model is trained using stochastic gradient descent, which randomly shuffles all known ratings at each epoch
    and goes through them.
    """

    def __init__(self, dimension: int, learning_rate: float, user_regularization: float = 0.0,
                 item_regularization: float = 0.0) -> None:
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

        super().__init__(dimension)

        if type(learning_rate) not in [float, np.float64]:
            raise TypeError('Learning rate should have float type')

        if learning_rate < 0:
            raise ValueError('Learning rate should be not lower than zero')

        if type(user_regularization) not in [float, np.float64]:
            raise TypeError('User_regularization should have float type')

        if user_regularization < 0:
            raise ValueError('User_regularization should be not lower than zero')

        if type(item_regularization) not in [float, np.float64]:
            raise TypeError('User_regularization should have float type')

        if item_regularization < 0:
            raise ValueError('User_regularization should be not lower than zero')

        self._rate: float = learning_rate
        self._user_regularization: float = user_regularization
        self._item_regularization: float = item_regularization

    def _calculate_delta(self, user_index: int, item_index: int, rating: float) -> float:
        """
        Method for calculate the difference between the original rating matrix and the matrix
        that was obtained at this point in time

        Parameters
        ----------
        user_index: int
            Index of current user
        item_index: int
            Index of current item
        rating: float
            Rating that the current user gave to the current product

        Returns
        -------
        Difference between original and calculated rating: float
        """

        # similarity between user and item
        similarity = self._users_matrix[user_index] @ self._items_matrix[item_index].T

        # get the difference between the true value of the rating and the approximate
        return rating - similarity

    def _calculate_users_matrix(self, user_index: int, item_index: int, delta: float) -> None:
        """
        Method for finding a row of users matrix

        Parameters
        ----------
        user_index: int
            Index of current user
        item_index: int
            Index of current item
        delta: float
            Difference between original and calculated rating
        """

        # the value of regularization for the user
        user_reg = self._user_regularization * np.sum(self._users_matrix[user_index]) / self._dimension
        # changing hidden variables for the user
        self._users_matrix[user_index] += self._rate * (delta * self._items_matrix[item_index] - user_reg)

    def _calculate_items_matrix(self, user_index: int, item_index: int, delta: float) -> None:
        """
        Method for finding a row of items matrix

        Parameters
        ----------
        user_index: int
            Index of current user
        item_index: int
            Index of current item
        delta: float
            Difference between original and calculated rating
        """

        # the value of regularization for the item
        item_reg = self._item_regularization * np.sum(self._items_matrix[item_index]) / self._dimension
        # changing hidden variables for the item
        self._items_matrix[item_index] += self._rate * (delta * self._users_matrix[user_index] - item_reg)

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        # determining known ratings
        self._users_indices: np.ndarray = data.row
        self._items_indices: np.ndarray = data.col
        self._ratings: np.ndarray = data.data

    def _train_one_epoch(self) -> None:
        # shuffle all data
        shuffle_indices = np.arange(self._ratings.shape[0])
        np.random.shuffle(shuffle_indices)

        for index in shuffle_indices:
            # get indices for user and item and rating
            user_index: int = self._users_indices[index]
            item_index: int = self._items_indices[index]
            rating: float = self._ratings[index]

            delta = self._calculate_delta(user_index, item_index, rating)
            self._calculate_users_matrix(user_index, item_index, delta)
            self._calculate_items_matrix(user_index, item_index, delta)

    def __str__(self) -> str:
        return f'SLFM [dimension = {self._dimension}, lr = {self._rate}, ' \
               f'user = {self._user_regularization}, item = {self._item_regularization}]'
