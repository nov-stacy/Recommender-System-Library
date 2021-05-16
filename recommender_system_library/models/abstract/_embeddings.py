import typing as tp

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from tqdm import tqdm

from recommender_system_library.metrics import mean_square_error, mean_absolute_error, root_mean_square_error
from recommender_system_library.models.abstract import AbstractRecommenderSystem


__all__ = ['EmbeddingsARS']


class EmbeddingsARS(AbstractRecommenderSystem, ABC):
    """
    Abstract class for recommender system that builds recommendations for the user based on the fact that all users and
    items are defined by vectors (categories of interests). For example, each component of such a vector can be
    interpreted as the degree of belonging of a given product to a certain category or
    the degree of interest of a given user in this category.
    Such vectors are representations that allow you to reduce entities into a single vector space.
    """

    class __EmbeddingDebug:
        """
        A class for determining the behavior if it is necessary to remember the error change on each epoch
        """

        __debugging_function_values = {
            'mse': mean_square_error,
            'mae': mean_absolute_error,
            'rmse': root_mean_square_error
        }

        def __init__(self) -> None:
            self.__debug_information: tp.Optional[tp.List[float]] = None  # array with errors on each epoch
            self.__debug_function = None  # error function

        def _update(self, debug_name: tp.Optional[str]) -> None:
            """
            Checking whether the debugger will be enabled

            Parameters
            ----------
            debug_name: str or None
                Name of the debugging function (mse, mae, rmse)
            """

            if debug_name is not None and type(debug_name) != str:
                raise TypeError('Debug name should have string type')

            if debug_name is not None and debug_name not in self.__debugging_function_values:
                raise ValueError(f'Debug should be in [{", ".join(self.__debugging_function_values.keys())}]')

            self.__debug_information = [] if debug_name else None
            self.__debug_function = self.__debugging_function_values[debug_name] if debug_name else None

        def _set(self, users_indices: np.ndarray, items_indices: np.ndarray, ratings: np.ndarray,
                  users_matrix: np.ndarray, items_matrix: np.ndarray) -> None:
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
            """

            if self.__debug_function is None:
                return

            if not (type(users_indices) == type(items_indices) == type(ratings) == type(users_matrix) ==
                    type(items_matrix) == np.ndarray):
                raise TypeError('All data should be numpy array type')

            data_shapes = [users_indices.shape, items_indices.shape, ratings.shape]
            matrix_shapes = [users_matrix.shape, items_matrix.shape]

            if not (len(data_shapes[0]) == len(data_shapes[1]) == len(data_shapes[2]) == 1):
                raise ValueError('Arrays should be 1D')

            if not (data_shapes[0] == data_shapes[1] == data_shapes[2]):
                raise ValueError('users_indices, items_indices and ratings should have same sizes')

            if not (len(matrix_shapes[0]) == len(matrix_shapes[1]) == 2):
                raise ValueError('Matrices should be 2D')

            if not (matrix_shapes[0][1] == matrix_shapes[1][1]):
                raise ValueError('users_matrix and items_matrix should have same dimension')

            true_ratings = list()
            predicted_ratings = list()

            # for each user and item for which the ratings are known
            for user_index, item_index, rating in zip(users_indices, items_indices, ratings):
                true_ratings.append(rating)
                predicted_ratings.append(users_matrix[user_index] @ items_matrix[item_index].T)

            self.__debug_information.append(
                self.__debug_function([np.array(true_ratings)], [np.array(predicted_ratings)]))

        def get(self) -> tp.List[float]:
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

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        if type(dimension) not in [int, np.int64]:
            raise TypeError('Dimension should have integer type')

        if dimension <= 0:
            raise ValueError('Dimension should be bigger than zero')

        self._dimension: int = dimension

        self._users_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self._items_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.debug_information = self.__EmbeddingDebug()

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

    def _create_information_for_debugging(self, data: sparse.coo_matrix, debug_name: tp.Optional[str]):
        """
        Method for determining the matrix of hidden features for the users and items

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        debug_name: str or None
            Name of the debugging function (mse, mae, rmse)
        """

        if debug_name:
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

    def _fit(self, epochs: int, debug_name: tp.Optional[str]) -> None:
        """
        Method for training a model

        Parameters
        ----------
        epochs: int
            The number of epochs that the method must pass
        debug_name: str or None
            Name of the debugging function (mse, mae, rmse)
        """

        for _ in tqdm(range(epochs)):
            self._train_one_epoch()
            if debug_name:
                self.debug_information._set(self._users_indices, self._items_indices, self._ratings,
                                            self._users_matrix, self._items_matrix)

    def fit(self, data: sparse.coo_matrix, epochs: int, debug_name: tp.Optional[str] = None) -> 'EmbeddingsARS':
        """
        Method for training a model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        debug_name: str or None
            Name of the debugging function (mse, mae, rmse)

        Returns
        -------
        Current instance of class: EmbeddingsRecommenderSystem
        """
        
        if type(data) != sparse.coo_matrix:
            raise TypeError('Data should be sparse matrix')
        
        if type(epochs) not in [int, np.int64]:
            raise TypeError('Number of epochs should have integer')

        if epochs <= 0:
            raise ValueError('Number of epochs should be positive')

        if debug_name is not None and type(debug_name) != str:
            raise TypeError('Debug name should have string type')

        self.debug_information._update(debug_name)
        self._create_user_items_matrix(data)
        self._create_information_for_debugging(data, debug_name)
        self._before_fit(data)
        self._fit(epochs, debug_name)
        self._is_trained = True

        return self

    def refit(self, data: sparse.coo_matrix, epochs: int = 100, debug_name: tp.Optional[str] = None) -> 'EmbeddingsARS':
        """
        Method for retrain model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass
        debug_name: str or None
            Name of the debugging function (mse, mae, rmse)

        Returns
        -------
        Current instance of class: EmbeddingsRecommenderSystem
        """

        self._check_trained_and_rise_error()

        if type(data) != sparse.coo_matrix:
            raise TypeError('Data should be sparse matrix')

        if type(epochs) not in [int, np.int64]:
            raise TypeError('Number of epochs should have integer')

        if epochs <= 0:
            raise ValueError('Number of epochs should be positive')

        if debug_name is not None and type(debug_name) != str:
            raise TypeError('Debug name should have string type')

        # add new users to matrix
        if data.shape[0] > self._users_count:
            matrix_part = np.random.randn(data.shape[0] - self._users_count, self._dimension)
            self._users_matrix = np.vstack((self._users_matrix, matrix_part))

        # add new items to matrix
        if data.shape[1] > self._items_count:
            matrix_part = np.random.randn(data.shape[1] - self._items_count, self._dimension)
            self._items_matrix = np.vstack((self._items_matrix, matrix_part))

        self.debug_information._update(debug_name)
        self._create_information_for_debugging(data, debug_name)
        self._before_fit(data)
        self._fit(epochs, debug_name)

        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:
        self._check_trained_and_rise_error()

        if type(user_index) not in [int, np.int64]:
            raise TypeError('Index should have integer type')

        if user_index < 0:
            raise ValueError('Index should be not negative')

        return np.nan_to_num(self._users_matrix[user_index] @ self._items_matrix.T)

    def predict(self, user_index: int) -> np.ndarray:
        self._check_trained_and_rise_error()

        if type(user_index) not in [int, np.int64]:
            raise TypeError('Index should have integer type')

        if user_index < 0:
            raise ValueError('Index should be not negative')

        return self.predict_ratings(user_index).argsort()[::-1]

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system with embeddings matrices for users and items'
