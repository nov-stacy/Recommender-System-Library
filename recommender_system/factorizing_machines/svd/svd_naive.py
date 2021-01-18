import numpy as np
import typing as tp
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from recommender_system.abstract import RecommenderSystem


class SingularValueDecompositionNaiveModel(RecommenderSystem):
    """
    Recommender system based on singular value decomposition.
    Naive realization: decompose the system matrix into matrices for users and items, facilitate them using
    approximation and give ranks using the product of a row with a user and a matrix of items
    """

    def __init__(self, dimension: tp.Optional[int] = None) -> None:
        """
        :param dimension: the number of singular values to keep (if dimension is None use original decomposition)
        """
        self.__dimension: tp.Optional[int] = dimension
        self.__first_matrix: np.array = np.array([])
        self.__second_matrix: np.array = np.array([])
        self.__singular_values: np.array = np.array([])
        self.__data: sparse.coo_matrix = sparse.eye(0)
        self.__indices_ptr: np.array = np.array([])
        self.__indices: np.array = np.array([])

    def __calculate_ratings(self, user_index: int) -> tp.List[tp.Tuple[int, int]]:
        # find items that user didnt mark
        known_ratings = self.__indices[self.__indices_ptr[user_index]: self.__indices_ptr[user_index + 1]]
        unknown_ratings = set(range(self.__data.shape[1])) - set(known_ratings)

        # all ratings for current user
        all_ratings_for_user = self.__first_matrix[user_index] @ np.diag(self.__singular_values) @ self.__second_matrix

        # predict ratings for items that user didnt mark
        predict_ratings = all_ratings_for_user[unknown_ratings]

        return list(zip(predict_ratings, unknown_ratings))

    def train(self, data: sparse.coo_matrix) -> 'SingularValueDecompositionNaiveModel':
        csr_data: sparse.csr_matrix = data.tocsr()
        self.__first_matrix, self.__singular_values, self.__second_matrix = svds(csr_data, k=self.__dimension)
        self.__indices_ptr = csr_data.indptr
        self.__indices: np.array = csr_data.indices
        self.__data = data.copy()
        return self

    def retrain(self, data: sparse.coo_matrix) -> 'SingularValueDecompositionNaiveModel':
        return self.train(data)

    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        ranked_list = self.__calculate_ratings(user_index)
        ranked_list.sort(reverse=True)
        return np.array(list(zip(*ranked_list[:k_items]))[1])

