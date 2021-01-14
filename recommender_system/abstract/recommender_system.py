from abc import ABC, abstractmethod
import numpy as np
import typing as tp


class RecommenderSystem(ABC):
    """
    Abstract class for system of recommendations
    """

    @abstractmethod
    def __calculate_ratings(self, user_index: int) -> tp.List[tp.Tuple[int, int]]:
        """
        Method to calculate ratings to items that user didnt mark
        :param user_index: the index of the user to make the prediction
        :return: list of elements of (rating, index_item) for each item
        """
        pass

    @abstractmethod
    def train(self, data: np.array) -> 'RecommenderSystem':
        """
        Method for training a model
        :param data: 2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
        :return: current instance of class
        """
        pass

    @abstractmethod
    def retrain(self, data: np.array) -> 'RecommenderSystem':
        """
        Method for retrain model
        :param data: 2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
        :return: current instance of class
        """
        pass

    @abstractmethod
    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        """
        Method for getting a ranked list of items that are recommended to the current user
        :param user_index: the index of the user to make the prediction
        :param k_items: quantity of items
        :return: list of items
        """
        pass
