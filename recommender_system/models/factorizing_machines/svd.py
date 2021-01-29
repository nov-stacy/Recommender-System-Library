import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from recommender_system.models.abstract_recommender_system import RecommenderSystem


class SingularValueDecompositionModel(RecommenderSystem):
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
        self.__dimension: int = dimension
        self.__first_matrix: np.ndarray = np.array([])
        self.__second_matrix: np.ndarray = np.array([])
        self.__singular_values: np.ndarray = np.array([])
        self.__data: sparse.coo_matrix = sparse.eye(0)
        self.__indices_ptr: np.ndarray = np.array([])
        self.__indices: np.ndarray = np.array([])

    def train(self, data: sparse.coo_matrix) -> 'SingularValueDecompositionModel':
        csr_data: sparse.csr_matrix = data.tocsr()
        self.__first_matrix, self.__singular_values, self.__second_matrix = svds(csr_data, k=self.__dimension)
        self.__indices_ptr = csr_data.indptr
        self.__indices = csr_data.indices
        self.__data = data.copy()
        return self

    def retrain(self, data: sparse.coo_matrix) -> 'SingularValueDecompositionModel':
        return self.train(data)

    def predict(self, user_index) -> np.ndarray:
        return self.__first_matrix[user_index] @ np.diag(self.__singular_values) @ self.__second_matrix

