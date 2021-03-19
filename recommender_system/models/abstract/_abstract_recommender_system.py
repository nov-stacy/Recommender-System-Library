from abc import ABC, abstractmethod

import numpy as np


class AbstractRecommenderSystem(ABC):
    """
    Abstract class for system of recommendations
    """

    @abstractmethod
    def train(self, *args) -> 'AbstractRecommenderSystem':
        """
        Method for training a model

        Returns
        -------
        Current instance of class: RecommenderSystem
        """

    @abstractmethod
    def retrain(self, *args) -> 'AbstractRecommenderSystem':
        """
        Method for retrain model

        Returns
        -------
        Current instance of class: RecommenderSystem
        """

    @abstractmethod
    def predict_ratings(self, *args) -> np.ndarray:
        """
        Method for getting a predicted ratings for current user

        Returns
        -------
        List of items: numpy array
        """

    @abstractmethod
    def predict(self, *args) -> np.ndarray:
        """
        Method for getting a predicted indices of items to user

        Returns
        -------
        List of indices: numpy array
        """

    @abstractmethod
    def __str__(self) -> str:
        return 'Abstract class for recommender system'
