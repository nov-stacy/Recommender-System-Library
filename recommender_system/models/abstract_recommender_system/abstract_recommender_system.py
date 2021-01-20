from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sparse


class RecommenderSystem(ABC):
    """
    Abstract class for system of recommendations
    """

    @abstractmethod
    def train(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        """
        Method for training a model
        :param data: 2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
        :return: current instance of class
        """
        pass

    @abstractmethod
    def retrain(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        """
        Method for retrain model
        :param data: 2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
        :return: current instance of class
        """
        pass

    @abstractmethod
    def predict(self, user_index: int) -> np.ndarray:
        """
        Method for getting a predicted ratings for current user
        :param user_index: the index of the user to make the prediction
        :return: list of items
        """
        pass
