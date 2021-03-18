import numpy as np
from scipy import sparse as sparse
from scipy.sparse.linalg import cg as solve
from tqdm import tqdm

from recommender_system.functional_errors.latent_error import calculate_error_for_latent_models
from recommender_system.models.abstract import RecommenderSystem, DebugInterface


class AlternatingLeastSquaresModel(RecommenderSystem, DebugInterface):
    """
    A model with hidden variables.

    Realization
    -----------
    Vectors denoting categories of interests are constructed for each user and item.
    Such vectors are representations that allow you to reduce entities into a single vector space.
    The model is trained due to the features of the functional. When fixing one of the matrices (for users or items),
    the functional becomes convex and can be found analytically.
    """

    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension: int
            The number of singular values to keep
        """

        super(DebugInterface, self).__init__(calculate_error_for_latent_models)

        self.__dimension: int = dimension

        self.__user_matrix: np.ndarray = np.array([])  # matrix of users with latent features
        self.__item_matrix: np.ndarray = np.array([])  # matrix of items with latent features

    def __calculate_user_matrix(self, data: sparse.coo_matrix, users_count: int, items_count: int) -> None:
        """
        Method for finding a matrix for users analytically

        Parameters
        ----------
        data: sparse matrix
            2-D matrix, where rows are users, and columns are items and at the intersection
            of a row and a column is the rating that this user has given to this item
        users_count: int
            Number of users in the system
        items_count: int
            Number of items in the system
        """

        # finding A for the equation Ax = B
        item_matrix = self.__item_matrix.T @ self.__item_matrix

        for user_index in range(users_count):

            # ratings that this user has set
            user_ratings = data.getrow(user_index).toarray().reshape((-1, items_count))[0]
            # weighted sum of latent features for items
            items_ratings = sum((user_ratings[index] * self.__item_matrix[index] for index in range(items_count)))

            self.__user_matrix[user_index] = solve(item_matrix, items_ratings)

    def __calculate_item_matrix(self, data: sparse.coo_matrix, users_count: int, items_count: int) -> None:
        """
        Method for finding a matrix for items analytically
        """

        # finding A for the equation Ax = B
        user_matrix = self.__item_matrix.T @ self.__item_matrix

        for item_index in range(items_count):

            # ratings that were given to this product
            item_ratings = data.getcol(item_index).toarray().reshape((-1, users_count))[0]
            # weighted sum of latent features for users
            users_ratings = sum((item_ratings[index] * self.__user_matrix[index] for index in range(users_count)))

            self.__item_matrix[item_index] = solve(user_matrix, users_ratings)

    def train(self, data: sparse.coo_matrix,
              epochs: int = 100,  # The number of epochs that the method must pass
              is_debug: bool = False  # Indicator of the need to maintain the error functionality on each epoch
              ) -> 'RecommenderSystem':

        users_count: int = data.shape[0]  # number of users in the system
        items_count: int = data.shape[1]  # number of items in the system

        # generate matrices with latent features
        self.__user_matrix: np.ndarray = np.random.randn(users_count, self.__dimension)
        self.__item_matrix: np.ndarray = np.random.randn(items_count, self.__dimension)

        # determining average values for users and items
        mean_users: np.ndarray = data.mean(axis=1)
        mean_items: np.ndarray = data.mean(axis=0).transpose()

        # determining known ratings
        users_indices: np.ndarray = data.row
        items_indices: np.ndarray = data.col
        ratings: np.ndarray = data.data

        self.__update_debug_information(is_debug)

        for _ in tqdm(range(epochs)):

            # calculate matrices for users and items analytically
            self.__calculate_user_matrix(data, users_count, items_count)
            self.__calculate_item_matrix(data, users_count, items_count)

            self.__set_debug_information(is_debug, self.__user_matrix, self.__item_matrix, mean_users, mean_items,
                                         users_indices, items_indices, ratings)

    def retrain(self, data: sparse.coo_matrix,
                epochs: int = 100,  # The number of epochs that the method must pass
                is_debug: bool = False  # Indicator of the need to maintain the error functionality on each epoch
                ) -> 'RecommenderSystem':
        self.train(data, epochs, is_debug)

    def predict_ratings(self, user_index) -> np.ndarray:
        return self.__user_matrix[user_index] @ self.__item_matrix.T
