import numpy as np
from scipy import sparse as sparse
import scipy.linalg as sla

from recommender_system_library.models.abstract import EmbeddingsRecommenderSystem


class ImplicitAlternatingLeastSquaresModel(EmbeddingsRecommenderSystem):

    def __init__(self, dimension: int, influence_regularization: float = 0) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super().__init__(dimension)

        self._influence: float = influence_regularization

    def _calculate_users_matrix(self, users_count):
        """
        Method for finding a matrix for users analytically

        Parameters
        ----------
        users_count: int
            Number of users in the system
        """
        quotient_matrix = self._items_matrix.T.dot(self._items_matrix)
        cho_decomposition = sla.cho_factor(quotient_matrix)

        for index in range(users_count):
            vector = self._items_matrix.T.dot(self._data.getrow(index).todense().T)
            self._users_matrix[index, :] = sla.cho_solve(cho_decomposition, vector).flatten()

    def _calculate_items_matrix(self, items_count):
        """
        Method for finding a matrix for items analytically

        Parameters
        ----------
        items_count: int
            Number of items in the system
        """

        quotient_matrix = self._users_matrix.T.dot(self._users_matrix)
        cho_decomposition = sla.cho_factor(quotient_matrix)

        for index in range(items_count):
            vector = self._users_matrix.T.dot(self._data.getcol(index).todense())
            self._items_matrix[index, :] = sla.cho_solve(cho_decomposition, vector).flatten()

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        matrix_ones = sparse.coo_matrix(np.ones(self._implicit_data.shape))

        self._data = data

        # determining the matrices of implicit data
        self._implicit_data: sparse.coo_matrix = (data != 0).astype(int)
        self._implicit_data_with_reg: sparse.coo_matrix = self._influence * matrix_ones + self._implicit_data

    def _train_one_epoch(self) -> None:
        # calculate matrices for users and items analytically
        self._calculate_users_matrix(self._users_count)
        self._calculate_items_matrix(self._items_count)

    def __str__(self) -> str:
        return f'iALS [dimension = {self._dimension}]'

