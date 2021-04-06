from scipy import sparse as sparse
from scipy.sparse.linalg import cg as solve

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

    def _calculate_user_matrix(self, data: sparse.coo_matrix, users_count: int, items_count: int) -> None:
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
        item_matrix = self._item_matrix.T @ self._item_matrix

        for user_index in range(users_count):

            # ratings that this user has set
            user_ratings = data.getrow(user_index).toarray().reshape((-1, items_count))[0]
            # weighted sum of latent features for items
            items_ratings = sum((user_ratings[index] * self._item_matrix[index] for index in range(items_count)))

            # solve Ax = B
            self._user_matrix[user_index] = solve(item_matrix, items_ratings)[0]

    def _calculate_item_matrix(self, data: sparse.coo_matrix, users_count: int, items_count: int) -> None:
        """
        Method for finding a matrix for items analytically

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
        user_matrix = self._item_matrix.T @ self._item_matrix

        for item_index in range(items_count):

            # ratings that were given to this product
            item_ratings = data.getcol(item_index).toarray().reshape((-1, users_count))[0]
            # weighted sum of latent features for users
            users_ratings = sum((item_ratings[index] * self._user_matrix[index] for index in range(users_count)))

            # solve Ax = B
            self._item_matrix[item_index] = solve(user_matrix, users_ratings)[0]

    def _before_fit(self, data: sparse.coo_matrix) -> None:
        self._data = data

    def _train_one_epoch(self) -> None:
        # calculate matrices for users and items analytically
        self._calculate_user_matrix(self._data, self._users_count, self._items_count)
        self._calculate_item_matrix(self._data, self._users_count, self._items_count)

    def __str__(self) -> str:
        return f'ALS [dimension = {self._dimension}]'
