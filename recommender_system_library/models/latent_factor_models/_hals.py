import numpy as np
from scipy import sparse as sparse

from recommender_system.models.abstract import EmbeddingsRecommenderSystem


class HierarchicalAlternatingLeastSquaresModel(EmbeddingsRecommenderSystem):
    """
    A model based only on the ratings.

    Realization
    -----------
    The model is trained due to the features of the functional.
    When fixing one of columns of matrices to find it analytically.
    """

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super().__init__(dimension)

    def _calculate_delta(self, index: int) -> np.array:
        """
        Method for calculate the difference between the original rating matrix and the matrix
        that was obtained at this point in time

        Parameters
        ----------
        index: int
            Index of the hidden attribute

        Returns
        -------
        Matrix of differences between original and calculated data: numpy array
        """

        indices = list(range(index)) + list(range(index + 1, self._dimension))
        calculated_data = sum((self._user_matrix[:, _index].reshape((self._user_matrix[:, _index].shape[0], 1)) @
                               self._item_matrix[:, _index].reshape((self._item_matrix[:, _index].shape[0], 1)).T
                               for _index in indices))
        return self._data - calculated_data

    def _calculate_user_matrix(self, index: int, delta: np.array) -> None:
        """
        Method for finding a column of users matrix

        Parameters
        ----------
        index: int
            Index of the hidden attribute
        delta: numpy array
            The difference between the original rating matrix and the matrix that was obtained at this point in time
        """

        denominator = self._item_matrix[:, index].T @ self._item_matrix[:, index]
        self._user_matrix[:, index] = delta @ self._item_matrix[:, index] / denominator

    def _calculate_item_matrix(self, index: int, delta: np.array) -> None:
        """
        Method for finding a column of items matrix

        Parameters
        ----------
        index: int
            Index of the hidden attribute
        delta: numpy array
            The difference between the original rating matrix and the matrix that was obtained at this point in time
        """

        denominator = self._user_matrix[:, index].T @ self._user_matrix[:, index]
        self._user_matrix[:, index] = self._user_matrix[:, index].T @ delta / denominator

    def _before_train(self, data: sparse.coo_matrix) -> None:
        self._data = data

    def _train_one_epoch(self) -> None:
        for index in range(self._dimension):
            # the difference between the original rating matrix and user_matrix @ item_matrix
            delta = self._calculate_delta(index)

            # calculate  a columns of matrices for users and items
            self._calculate_user_matrix(index, delta)
            self._calculate_item_matrix(index, delta)

    def __str__(self) -> str:
        return f'HALS [dimension = {self._dimension}]'
