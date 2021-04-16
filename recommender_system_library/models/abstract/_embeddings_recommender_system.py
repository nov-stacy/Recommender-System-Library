import typing as tp

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from tqdm import tqdm

from recommender_system_library.models.abstract import AbstractRecommenderSystem


__all__ = ['EmbeddingsRecommenderSystem', 'EmbeddingDebug']


class EmbeddingDebug:
    """
    A class for determining the behavior if it is necessary to remember the error change on each epoch
    """

    def __init__(self) -> None:
        self._debug_information: tp.Optional[tp.List[float]] = None  # array with errors on each epoch

        # matrices with latent features
        self._users_matrix: np.ndarray = np.array([])
        self._items_matrix: np.ndarray = np.array([])

    def update(self, is_debug: bool) -> None:
        """
        Checking whether the debugger will be enabled

        Parameters
        ----------
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch
        """

        if type(is_debug) != bool:
            raise TypeError('Debug indices should have bool type')

        self._debug_information = [] if is_debug else None

    def set(self, users_indices: np.ndarray, items_indices: np.ndarray, ratings: np.ndarray,
            users_matrix: np.ndarray, items_matrix: np.ndarray,
            mean_users: np.ndarray, mean_items: np.ndarray) -> None:
        """
        Calculate functional of error

        Parameters
        ----------
        users_indices: numpy array
            Array for indices with not null ratings
        items_indices: numpy array
            Array for indices with not null ratings
        ratings: numpy array
            Array with not null ratings
        users_matrix: numpy array
            Matrix of users with latent features
        items_matrix: numpy array
            Matrix of items with latent features
        mean_users: numpy array
            Average values for users ratings
        mean_items: numpy array
            Average values for items ratings
        """

        if not(type(users_indices) == type(items_indices) == type(ratings) == type(users_matrix) ==
               type(items_matrix) == type(mean_users) == type(mean_items) == np.ndarray):
            raise TypeError('All data should be numpy array type')

        data_shapes = [users_indices.shape, items_indices.shape, ratings.shape]
        matrix_shapes = [users_matrix.shape, items_matrix.shape]
        mean_shapes = [mean_users.shape, mean_items.shape]

        if not(len(data_shapes[0]) == len(data_shapes[1]) == len(data_shapes[2]) == 1):
            raise ValueError('Arrays should be 1D')

        if not(data_shapes[0] == data_shapes[1] == data_shapes[2]):
            raise ValueError('users_indices, items_indices and ratings should have same sizes')

        if not(len(matrix_shapes[0]) == len(matrix_shapes[1]) == 2):
            raise ValueError('Matrices should be 2D')

        if not(matrix_shapes[0][1] == matrix_shapes[1][1]):
            raise ValueError('users_matrix and items_matrix should have same dimension')

        if not(len(mean_shapes[0]) == len(mean_shapes[1]) == 2) or not(mean_shapes[0][1] == mean_shapes[1][1] == 1):
            raise ValueError('Means should be 2D: (1, N)')

        if not(mean_shapes[0][0] == matrix_shapes[0][0] and mean_shapes[1][0] == matrix_shapes[1][0]):
            raise ValueError('Means and matrices should have same sizes')

        result: float = 0  # error functionality

        # for each user and item for which the ratings are known
        for user_index, item_index, rating in zip(users_indices, items_indices, ratings):
            # similarity between user and item
            similarity = users_matrix[user_index] @ items_matrix[item_index].T
            # adding to the functionality
            result += (rating - mean_users[user_index] - mean_items[item_index] - similarity) ** 2

        self._debug_information.append(np.asscalar(result))

    def get(self) -> tp.List[float]:
        """
        Method for getting a list of functionality errors that were optimized during training

        Returns
        -------
        list of functionality errors: list[float]
        """

        if self._debug_information is None:
            raise AttributeError("No debug information because there was is_debug = False during training")
        else:
            return self._debug_information


class EmbeddingsRecommenderSystem(AbstractRecommenderSystem, ABC):
    """
    Abstract class for recommender system that builds recommendations for the user based on the fact that all users and
    items are defined by vectors (categories of interests). For example, each component of such a vector can be
    interpreted as the degree of belonging of a given product to a certain category or
    the degree of interest of a given user in this category.
    Such vectors are representations that allow you to reduce entities into a single vector space.
    """

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        if type(dimension) != int:
            raise TypeError('Dimension should have integer type')

        if dimension <= 0:
            raise ValueError('Dimension should be bigger than zero')

        self._dimension: int = dimension

        self._users_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self._items_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.debug_information = EmbeddingDebug()

    def _create_user_items_matrix(self, data: sparse.coo_matrix) -> None:
        """
        Method for determining the matrix of hidden features for the users and items

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        """

        # determining the number of users and items
        self._users_count: int = data.shape[0]
        self._items_count: int = data.shape[1]

        # generate matrices with latent features
        self._users_matrix: np.ndarray = np.random.randn(self._users_count, self._dimension)
        self._items_matrix: np.ndarray = np.random.randn(self._items_count, self._dimension)

        # determining average values for users and items
        self._mean_users: np.ndarray = np.array(data.mean(axis=1))
        self._mean_items: np.ndarray = np.array(data.mean(axis=0).transpose())

    def _create_information_for_debugging(self, data: sparse.coo_matrix, is_debug: bool):
        """
        Method for determining the matrix of hidden features for the users and items

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch
        """

        if is_debug:
            # determining known ratings
            self._users_indices: tp.Optional[np.ndarray] = data.row
            self._items_indices: tp.Optional[np.ndarray] = data.col
            self._ratings: tp.Optional[np.ndarray] = data.data
        else:
            self._users_indices: tp.Optional[np.ndarray] = None
            self._items_indices: tp.Optional[np.ndarray] = None
            self._ratings: tp.Optional[np.ndarray] = None

    @abstractmethod
    def _before_fit(self, data: sparse.coo_matrix) -> None:
        """
        Method that is called before training starts to save all the data that is used during training

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        """

    @abstractmethod
    def _train_one_epoch(self) -> None:
        """
        Method for teaching a method during a single epoch
        """

    def _fit(self, epochs: int, is_debug: bool = False) -> None:
        """
        Method for training a model

        Parameters
        ----------
        epochs: int
            The number of epochs that the method must pass
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch
        """

        for _ in tqdm(range(epochs)):
            self._train_one_epoch()
            if is_debug:
                self.debug_information.set(self._users_indices, self._items_indices, self._ratings,
                                           self._users_matrix, self._items_matrix, self._mean_users, self._mean_items)

    def fit(self, data: sparse.coo_matrix, epochs: int, is_debug: bool = False) -> 'EmbeddingsRecommenderSystem':
        """
        Method for training a model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch

        Returns
        -------
        Current instance of class: EmbeddingsRecommenderSystem
        """
        
        if type(data) != sparse.coo_matrix:
            raise TypeError('Data should be sparse matrix')
        
        if type(epochs) != int:
            raise TypeError('Number of epochs should have integer')

        if epochs <= 0:
            raise ValueError('Number of epochs should be positive')

        if type(is_debug) != bool:
            raise TypeError('Indicator of debug should have bool type')

        self.debug_information.update(is_debug)
        self._create_user_items_matrix(data)
        self._create_information_for_debugging(data, is_debug)
        self._before_fit(data)
        self._fit(epochs, is_debug)
        self._is_trained = True

        return self

    def refit(self, data: sparse.coo_matrix,
              epochs: int = 100, is_debug: bool = False) -> 'EmbeddingsRecommenderSystem':
        """
        Method for retrain model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch

        Returns
        -------
        Current instance of class: EmbeddingsRecommenderSystem
        """

        self._check_trained_and_rise_error()

        if type(data) != sparse.coo_matrix:
            raise TypeError('Data should be sparse matrix')

        if type(epochs) != int:
            raise TypeError('Number of epochs should have integer')

        if epochs <= 0:
            raise ValueError('Number of epochs should be positive')

        if type(is_debug) != bool:
            raise TypeError('Indicator of debug should have bool type')

        # add new users to matrix
        if data.shape[0] > self._users_count:
            matrix_part = np.random.randn(data.shape[0] - self._users_count, self._dimension)
            self._users_matrix = np.vstack((self._users_matrix, matrix_part))

        # add new items to matrix
        if data.shape[1] > self._items_count:
            matrix_part = np.random.randn(data.shape[1] - self._items_count, self._dimension)
            self._items_matrix = np.vstack((self._items_matrix, matrix_part))

        self.debug_information.update(is_debug)
        self._create_information_for_debugging(data, is_debug)
        self._before_fit(data)
        self._fit(epochs)

        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:
        """
        Method for getting a predicted ratings for current user

        Parameters
        ----------
        user_index: int
            The index of the user to make the prediction

        Returns
        -------
        List of items: numpy array
        """

        self._check_trained_and_rise_error()

        if type(user_index) != int:
            raise TypeError('Index should have integer type')

        if user_index < 0:
            raise ValueError('Index should be not negative')

        return self._users_matrix[user_index] @ self._items_matrix.T

    def predict(self, user_index: int, items_count: int) -> np.ndarray:
        """
        Method for getting a predicted indices of items to user

        Parameters
        ----------
        user_index: int
            The index of the user to make the prediction
        items_count: int
            The count of items to predict

        Returns
        -------
        List of indices: numpy array
        """

        self._check_trained_and_rise_error()

        if type(user_index) != int:
            raise TypeError('Index should have integer type')

        if user_index < 0:
            raise ValueError('Index should be not negative')

        if type(items_count) != int:
            raise TypeError('Count of items should have integer type')

        if items_count <= 0:
            raise ValueError('Count of items should be positive')

        return self.predict_ratings(user_index).argsort()[::-1][:items_count]

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system with embeddings matrices for users and items'
