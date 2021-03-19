import numpy as np
from scipy import sparse as sparse

from recommender_system.models.abstract import EmbeddingsRecommenderSystem


class StochasticImplicitLatentFactorModel(EmbeddingsRecommenderSystem):
    """
    A model based on the fact that any signals from the user are better than their absence.

    Realization
    -----------
    The model is trained using stochastic gradient descent, which randomly shuffles all known ratings at each epoch
    and goes through them.
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

        super().__init__(dimension)

        self.__rate: float = learning_rate
        self.__influence: float = influence_regularization
        self.__user_regularization: float = user_regularization
        self.__item_regularization: float = item_regularization

    def __calculate_delta(self, user_index: int, item_index: int, rating: float, indicator: int) -> float:
        """
        Method for calculate the difference between the original rating matrix and the matrix
        that was obtained at this point in time

        Parameters
        ----------
        user_index: int
            Index of current user
        item_index: int
            Index of current item
        rating: int
            All ratings that user gave
        indicator: numpy array
            Indicators of implicit user interest in items

        Returns
        -------
        Difference between original and calculated ratings: float
        """

        # similarity between user and item
        similarity = self._user_matrix[user_index] @ self._item_matrix[item_index].T

        # get the difference between the true value of the rating and the approximate
        return (indicator - np.asscalar(self._mean_users[user_index]) -
                np.asscalar(self._mean_items[item_index]) - similarity) * (1 + self.__influence * rating)

    def __calculate_user_matrix(self, user_index: int, item_index: int, delta: float) -> None:
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
        user_reg = self.__user_regularization * np.sum(self._user_matrix[user_index]) / self._dimension
        # changing hidden variables for the user
        self._user_matrix[user_index] += self.__rate * (delta * self._item_matrix[item_index] - user_reg)

    def __calculate_item_matrix(self, user_index: int, item_index: int, delta: float) -> None:
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
        # the value of regularization for the item
        item_reg = self.__item_regularization * np.sum(self._item_matrix[item_index]) / self._dimension
        # changing hidden variables for the item
        self._item_matrix[item_index] += self.__rate * (delta * self._user_matrix[user_index] - item_reg)

    def _before_train(self, data: sparse.coo_matrix) -> None:
        self._data: sparse.coo_matrix = data

        # determining the average values of implicit interest for users and items
        self._mean_users: np.ndarray = (data != 0).astype(int).mean(axis=1)
        self._mean_items: np.ndarray = (data != 0).mean(axis=0).transpose()

    def _train_one_epoch(self) -> None:
        # random users
        users_indices = np.arange(self._users_count)
        np.random.shuffle(users_indices)

        for user_index in users_indices:

            # random items
            items_indices = np.arange(self._items_count)
            np.random.shuffle(items_indices)

            # ratings that the user has set for items
            user_ratings: np.ndarray = self._data.getrow(user_index).toarray()[0]
            # indicators of implicit user interest in items
            ii_user: np.ndarray = (user_ratings != 0).astype(int)

            for item_index in items_indices:
                delta = self.__calculate_delta(user_index, item_index, user_ratings[item_index], ii_user[item_index])
                self.__calculate_user_matrix(user_index, item_index, delta)
                self.__calculate_item_matrix(user_index, item_index, delta)

    def __str__(self) -> str:
        return f'iSGD [dimension = {self._dimension}]'