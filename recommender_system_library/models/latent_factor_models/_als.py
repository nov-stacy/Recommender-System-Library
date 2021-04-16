from scipy import sparse as sparse
import scipy.linalg as sla

from recommender_system_library.models.abstract import EmbeddingsRecommenderSystem


class AlternatingLeastSquaresModel(EmbeddingsRecommenderSystem):
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

    def _calculate_user_matrix(self, users_count: int) -> None:
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
            self._users_matrix[index, :] = \
                sla.cho_solve(cho_decomposition,
                              self._items_matrix.T.dot(self._data.getrow(index).todense().T)).flatten()

    def _calculate_item_matrix(self, items_count: int) -> None:
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
            self._items_matrix[index, :] = \
                sla.cho_solve(cho_decomposition,
                              self._users_matrix.T.dot(self._data.getcol(index).todense())).flatten()

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        self._data = data

    def _train_one_epoch(self) -> None:
        # calculate matrices for users and items analytically
        self._calculate_user_matrix(self._users_count)
        self._calculate_item_matrix(self._items_count)

    def __str__(self) -> str:
        return f'ALS [dimension = {self._dimension}]'
