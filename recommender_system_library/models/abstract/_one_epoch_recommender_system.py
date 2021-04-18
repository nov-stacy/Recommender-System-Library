from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from recommender_system_library.models.abstract import AbstractRecommenderSystem


class AbstractRecommenderSystemTrainWithOneEpoch(AbstractRecommenderSystem, ABC):
    """
    Abstract class for recommender system which are trained with the help of a single epoch and
    don`t have the possibility of retraining
    """

    @abstractmethod
    def _fit(self, data: sparse.coo_matrix) -> None:
        """
        Method for training a model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item

        Returns
        -------
        Current instance of class : RecommenderSystem
        """

    def fit(self, data: sparse.coo_matrix) -> 'AbstractRecommenderSystemTrainWithOneEpoch':
        """
        Method for training a model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item

        Returns
        -------
        Current instance of class : RecommenderSystem
        """

        if type(data) != sparse.coo_matrix:
            raise TypeError('Data should be sparse matrix')

        self._fit(data)
        self._is_trained = True

        return self

    def refit(self, data: sparse.coo_matrix) -> 'AbstractRecommenderSystemTrainWithOneEpoch':
        """
        Method for retrain model

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item

        Returns
        -------
        Current instance of class: RecommenderSystem
        """

        self._check_trained_and_rise_error()

        if type(data) != sparse.coo_matrix:
            raise TypeError('Data should be sparse matrix')

        return self.fit(data)

    @abstractmethod
    def _predict_ratings(self, user_index: int) -> np.ndarray:
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

        if type(user_index) not in [int, np.int64]:
            raise TypeError('Index should have integer type')

        if user_index < 0:
            raise ValueError('Index should be not negative')

        return self._predict_ratings(user_index)

    def _predict(self, user_index: int, items_count: int):
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
        list of indices: numpy array
        """
        return self.predict_ratings(user_index).argsort()[::-1][:items_count]

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
        list of indices: numpy array
        """

        self._check_trained_and_rise_error()

        if type(user_index) not in [int, np.int64]:
            raise TypeError('Index should have integer type')

        if user_index < 0:
            raise ValueError('Index should be positive and less than count of users')

        if type(items_count) not in [int, np.int64]:
            raise TypeError('Count of items should have integer type')

        if items_count <= 0:
            raise ValueError('Count of items should be positive')

        return self._predict(user_index, items_count)

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system with train in one epoch'
