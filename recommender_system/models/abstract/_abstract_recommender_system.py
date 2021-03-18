from abc import ABC, abstractmethod
from recommender_system.extra_functions.predict_values_parsers import calculate_issue_ranked_list
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

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item

        Returns
        -------
        Current instance of class : RecommenderSystem
        """

    def retrain(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
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
    def predict_ratings(self, user_index) -> np.ndarray:
        """
        Method for getting a predicted ratings for current user

        Parameters
        ----------
        user_index: int
            The index of the user to make the prediction

        Returns
        -------
        list of items: numpy array
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
        return calculate_issue_ranked_list(self.predict_ratings(user_index), k_items=items_count)
