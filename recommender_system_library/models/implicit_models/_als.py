from scipy import sparse as sparse
import scipy.linalg as sla

from recommender_system_library.models.abstract import EmbeddingsRecommenderSystem


class ImplicitAlternatingLeastSquaresModel(EmbeddingsRecommenderSystem):

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super().__init__(dimension)

    def _calculate_user_matrix(self, users_count):
        pass

    def _calculate_item_matrix(self, items_count):
        pass

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        self._data = data

    def _train_one_epoch(self) -> None:
        # calculate matrices for users and items analytically
        self._calculate_user_matrix(self._users_count)
        self._calculate_item_matrix(self._items_count)

    def __str__(self) -> str:
        return f'iALS [dimension = {self._dimension}]'

