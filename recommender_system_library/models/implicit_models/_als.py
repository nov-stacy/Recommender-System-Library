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

        for index in range(users_count):
            ith_row = self._implicit_data_with_reg.getrow(index).toarray()
            quotient_matrix = (ith_row * self._items_matrix.T).dot(self._items_matrix)
            self._users_matrix[index, :] = \
                sla.solve(quotient_matrix,
                          self._items_matrix.T.dot(ith_row.T * self._implicit_data.getrow(index).toarray().T)).flatten()

    def _calculate_items_matrix(self, items_count):
        """
        Method for finding a matrix for items analytically

        Parameters
        ----------
        items_count: int
            Number of items in the system
        """

        for index in range(items_count):
            ith_row = self._implicit_data_with_reg.getrow(index).toarray()
            ith_col = self._implicit_data_with_reg.getcol(index).toarray()
            quotient_matrix = (ith_row * self._items_matrix.T).dot(self._users_matrix)
            self._items_matrix[index, :] = \
                sla.solve(quotient_matrix,
                          self._users_matrix.T.dot(ith_col * self._implicit_data.getcol(index).toarray())).flatten()

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        self._data = data

        matrix_ones = sparse.coo_matrix(np.ones(data.shape))

        # determining the matrices of implicit data
        self._implicit_data: sparse.coo_matrix = (data != 0).astype(int)
        self._implicit_data_with_reg: sparse.coo_matrix = self._influence * matrix_ones + self._implicit_data

    def _train_one_epoch(self) -> None:
        # calculate matrices for users and items analytically
        self._calculate_users_matrix(self._users_count)
        self._calculate_items_matrix(self._items_count)

    def __str__(self) -> str:
        return f'iALS [dimension = {self._dimension}]'

