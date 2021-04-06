from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from recommender_system.models.abstract import AbstractRecommenderSystem


class OneEpochAbstractRecommenderSystem(AbstractRecommenderSystem, ABC):
    """
    Abstract class for recommender system which are trained with the help of a single epoch and
    don`t have the possibility of retraining
    """

    @abstractmethod
    def train(self, data: sparse.coo_matrix) -> 'OneEpochAbstractRecommenderSystem':
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

    def retrain(self, data: sparse.coo_matrix) -> 'OneEpochAbstractRecommenderSystem':
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

        return self.train(data)

    @abstractmethod
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

        return self.predict_ratings(user_index).argsort()[::-1][:items_count]

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system with train in one epoch'
