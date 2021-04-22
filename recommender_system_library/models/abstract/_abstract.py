from abc import ABC, abstractmethod

import numpy as np


__all__ = ['AbstractRecommenderSystem']


class AbstractRecommenderSystem(ABC):
    """
    Abstract class for system of recommendations
    """

    _is_trained = False  # indicator for determining whether the model has been trained

    @property
    def is_trained(self):
        """
        Indicator for determining whether the model has been trained
        """
        return self._is_trained

    def _check_trained_and_rise_error(self):
        """
        Indicator for determining whether the model has been trained
        """
        if not self._is_trained:
            raise AttributeError('Model should be trained')

    @abstractmethod
    def fit(self, *args, **kwargs) -> 'AbstractRecommenderSystem':
        """
        Method for training a model

        Returns
        -------
        Current instance of class: RecommenderSystem
        """

    @abstractmethod
    def refit(self, *args, **kwargs) -> 'AbstractRecommenderSystem':
        """
        Method for retrain model

        Returns
        -------
        Current instance of class: RecommenderSystem
        """

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

    @abstractmethod
    def predict(self, user_index: int) -> np.ndarray:
        """
        Method for getting a predicted indices of items to user

        Parameters
        ----------
        user_index: int
            The index of the user to make the prediction

        Returns
        -------
        List of indices: numpy array
        """

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system'
