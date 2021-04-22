import numpy as np
from scipy import sparse as sparse
import scipy.linalg as sla

from recommender_system_library.models.abstract import EmbeddingsARS


class AlternatingLeastSquaresModel(EmbeddingsARS):
    """
    A model based only on the ratings.

    Realization
    -----------
    The model is trained due to the features of the functional. When fixing one of the matrices (for users or items),
    the functional becomes convex and can be found analytically.
    """

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super().__init__(dimension)

    def _calculate_users_matrix(self) -> None:
        """
        Method for finding a matrix for users analytically
        """

        quotient_matrix = self._items_matrix.T.dot(self._items_matrix)
        try:
            cho_decomposition = sla.cho_factor(quotient_matrix)
            for index in range(self._users_matrix.shape[0]):
                self._users_matrix[index, :] = sla.cho_solve(cho_decomposition, self._items_matrix.T.dot(self._data.getrow(index).todense().T)).flatten()
        except (np.linalg.LinAlgError, sla.LinAlgError):
            pinv_result = np.linalg.pinv(quotient_matrix)
            for index in range(self._users_matrix.shape[0]):
                self._users_matrix[index, :] = pinv_result.dot(self._items_matrix.T.dot(self._data.getrow(index).todense().T)).flatten()

    def _calculate_items_matrix(self) -> None:
        """
        Method for finding a matrix for items analytically
        """

        quotient_matrix = self._users_matrix.T.dot(self._users_matrix)
        try:
            cho_decomposition = sla.cho_factor(quotient_matrix)
            for index in range(self._items_matrix.shape[0]):
                self._items_matrix[index, :] = sla.cho_solve(cho_decomposition, self._users_matrix.T.dot(self._data.getcol(index).todense())).flatten()
        except (np.linalg.LinAlgError, sla.LinAlgError):
            pinv_result = np.linalg.pinv(quotient_matrix)
            for index in range(self._users_matrix.shape[0]):
                self._users_matrix[index, :] = pinv_result.dot(self._items_matrix.T.dot(self._data.getrow(index).todense().T)).flatten()

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        self._data = data

    def _train_one_epoch(self) -> None:
        # calculate matrices for users and items analytically
        self._calculate_users_matrix()
        self._calculate_items_matrix()

    def __str__(self) -> str:
        return f'ALS [dimension = {self._dimension}]'
