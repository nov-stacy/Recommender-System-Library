from abc import ABC, abstractmethod

import numpy as np


class AbstractRecommenderSystem(ABC):
    """
    Abstract class for system of recommendations
    """

    _is_trained = False

    @property
    def is_trained(self):
        return self._is_trained

    def _is_predict(self):
        if not self._is_trained:
            raise RuntimeError('Model should be trained')

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

        Returns
        -------
        List of items: numpy array
        """

    @abstractmethod
    def predict(self, user_index: int, items_count: int) -> np.ndarray:
        """
        Method for getting a predicted indices of items to user

        Returns
        -------
        List of indices: numpy array
        """

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system'
