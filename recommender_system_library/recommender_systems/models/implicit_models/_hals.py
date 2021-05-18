import numpy as np
from scipy import sparse as sparse

from recommender_systems.models.abstract import EmbeddingsARS


class ImplicitHierarchicalAlternatingLeastSquaresModel(EmbeddingsARS):
    """
    A model based only on the ratings.

    Realization
    -----------
    The model is trained due to the features of the functional.
    When fixing one of columns of matrices to find it analytically.
    """

    def __init__(self, dimension: int, influence_regularization: float = 0) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super().__init__(dimension)

        self._influence: float = influence_regularization

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

        data = self._implicit_data.multiply(self._implicit_data_with_reg)
        indices = list(range(index)) + list(range(index + 1, self._dimension))
        calculated_data = np.sum(self._users_matrix[:, _index].reshape((self._users_matrix[:, _index].shape[0], 1)) @
                                 self._items_matrix[:, _index].reshape((self._items_matrix[:, _index].shape[0], 1)).T
                                 for _index in indices)
        result = self._implicit_data_with_reg.multiply(calculated_data)
        return data - result

    def _calculate_users_matrix(self, index: int, delta: np.array) -> None:
        """
        Method for finding a column of users matrix

        Parameters
        ----------
        index: int
            Index of the hidden attribute
        delta: numpy array
            The difference between the original rating matrix and the matrix that was obtained at this point in time
        """

        denominator = self._items_matrix[:, index].T @ self._items_matrix[:, index]
        self._users_matrix[:, index] = delta @ self._items_matrix[:, index] / denominator

    def _calculate_items_matrix(self, index: int, delta: np.array) -> None:
        """
        Method for finding a column of items matrix

        Parameters
        ----------
        index: int
            Index of the hidden attribute
        delta: numpy array
            The difference between the original rating matrix and the matrix that was obtained at this point in time
        """

        denominator = self._users_matrix[:, index].T @ self._users_matrix[:, index]
        self._items_matrix[:, index] = delta.T @ self._users_matrix[:, index] / denominator

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        matrix_ones = sparse.coo_matrix(np.ones(data.shape))

        # determining the matrices of implicit data
        self._implicit_data: sparse.coo_matrix = (data != 0).astype(int).tocoo()
        self._implicit_data_with_reg: sparse.coo_matrix = (self._influence * matrix_ones + self._implicit_data).tocoo()

    def _train_one_epoch(self) -> None:
        for index in range(self._dimension):
            # the difference between the original rating matrix and users_matrix @ items_matrix
            delta = self._calculate_delta(index)
            # calculate  a columns of matrices for users and items
            self._calculate_users_matrix(index, delta)
            self._calculate_items_matrix(index, delta)

    def __str__(self) -> str:
        return f'iHALS [dimension = {self._dimension}, influence = {self._influence}]'
