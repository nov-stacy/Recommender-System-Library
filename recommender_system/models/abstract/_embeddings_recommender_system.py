import typing as tp

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from tqdm import tqdm

from recommender_system.models.abstract import AbstractRecommenderSystem


class EmbeddingDebug:

    def __init__(self) -> None:
        self.__is_debug = False
        self.__debug_information: tp.Optional[tp.List[float]] = None  # array with errors on each epoch

    def update(self, is_debug: bool) -> None:
        """
        Checking whether the debugger will be enabled

        Parameters
        ----------
        is_debug: bool
            Indicator of the need to maintain the error functionality on each epoch
        """
        self.__debug_information = [] if is_debug else None
        self.__is_debug = is_debug

    def set(self, model: 'EmbeddingsRecommenderSystem') -> None:
        """
        Calculate functional of error

        Parameters
        ----------
        model: EmbeddingsRecommenderSystem
            the model to find for
        """
        if not self.__is_debug:
            return

        result: float = 0  # error functionality

        # for each user and item for which the ratings are known
        for user_index, item_index, rating in zip(model._users_indices, model._items_indices, model._ratings):
            # similarity between user and item
            similarity = model._user_matrix[user_index] @ model._item_matrix[item_index].T
            # adding to the functionality
            result += (rating - model._mean_users[user_index] - model._mean_items[item_index] - similarity) ** 2

        self.__debug_information.append(np.asscalar(result))

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


class EmbeddingsRecommenderSystem(AbstractRecommenderSystem, ABC):
    """
    Abstract class for recommender system that builds recommendations for the user based on the fact that all users and
    items are defined by vectors (categories of interests). For experiments, each component of such a vector can be
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

        self._dimension: int = dimension

        self._user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self._item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

        self.debug_information = EmbeddingDebug()

    def _create_user_item_matrix(self, data: sparse.coo_matrix) -> None:
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
        self._user_matrix: np.ndarray = np.random.randn(self._users_count, self._dimension)
        self._item_matrix: np.ndarray = np.random.randn(self._items_count, self._dimension)

        # determining average values for users and items
        self._mean_users: np.ndarray = data.mean(axis=1)
        self._mean_items: np.ndarray = data.mean(axis=0).transpose()

        # determining known ratings
        self._users_indices: np.ndarray = data.row
        self._items_indices: np.ndarray = data.col
        self._ratings: np.ndarray = data.data

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

        return self._user_matrix[user_index] @ self._item_matrix.T

    @abstractmethod
    def _before_train(self, data: sparse.coo_matrix) -> None:
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

    def _train(self, epochs: int) -> None:
        """
        Method for training a model

        Parameters
        ----------
        epochs: int
            The number of epochs that the method must pass
        """

        for _ in tqdm(range(epochs)):
            self._train_one_epoch()
            self.debug_information.set(self)

    def train(self, data: sparse.coo_matrix, epochs: int, is_debug: bool = False) -> 'EmbeddingsRecommenderSystem':
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

        self.debug_information.update(is_debug)
        self._create_user_item_matrix(data)
        self._before_train(data)
        self._train(epochs)

        return self

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

        return self.predict_ratings(user_index).argsort()[::-1][:items_count]

    def retrain(self, data: sparse.coo_matrix, epochs: int = 100) -> 'EmbeddingsRecommenderSystem':
        """
        Method for retrain model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        epochs: int
            The number of epochs that the method must pass

        Returns
        -------
        Current instance of class: EmbeddingsRecommenderSystem
        """

        # add new users to matrix
        if data.shape[0] > self._users_count:
            matrix_part = np.random.randn(data.shape[0] - self._users_count, self._dimension)
            self._user_matrix = np.vstack((self._user_matrix, matrix_part))

        # add new items to matrix
        if data.shape[1] > self._items_count:
            matrix_part = np.random.randn(data.shape[1] - self._items_count, self._dimension)
            self._item_matrix = np.vstack((self._item_matrix, matrix_part))

        self._before_train(data)
        self._train(epochs)

        return self

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system with embeddings matrices for users and items'
