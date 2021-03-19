import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from recommender_system.models.abstract import OneEpochAbstractRecommenderSystem


class SingularValueDecompositionModel(OneEpochAbstractRecommenderSystem):
    """
    Recommender system based on singular value decomposition.

    Realization
    -----------
    Decompose the system matrix into matrices for users and items, facilitate them using
    approximation and give ranks using the product of a row with a user and a matrix of items
    """

    def __init__(self, dimension: int) -> None:
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """
        self._dimension: int = dimension
        self._first_matrix: np.ndarray = np.array([])
        self._second_matrix: np.ndarray = np.array([])
        self._singular_values: np.ndarray = np.array([])

    def predict_ratings(self, user_index) -> np.ndarray:
        return self._first_matrix[user_index] @ np.diag(self._singular_values) @ self._second_matrix

    def train(self, data: sparse.coo_matrix) -> 'SingularValueDecompositionModel':
        self._first_matrix, self._singular_values, self._second_matrix = svds(data.tocsr(), k=self._dimension)
        return self

    def __str__(self) -> str:
        return f'SVD [dimension = {self._dimension}]'
