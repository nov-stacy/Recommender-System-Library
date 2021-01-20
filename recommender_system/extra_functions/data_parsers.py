import scipy.sparse as sparse
import numpy as np
import typing as tp


def calculate_unknown_ratings(user_ratings: sparse.spmatrix) -> np.ndarray:
    """
    Method to get items that user didnt mark
    :param user_ratings: row of ratings that below to user
    :return: indices of items
    """

    if len(user_ratings.shape) != 2 or user_ratings.shape[0] != 1:
        raise ValueError('Data needs to be 1D array')

    # calculate indices of marked items
    csr_data: sparse.csr_matrix = sparse.csr_matrix(user_ratings)
    known_ratings: np.ndarray = csr_data.indices

    return np.array(set(range(csr_data.shape[1])) - set(known_ratings))


def train_test_split(matrix: sparse.coo_matrix) -> tp.Tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """
    Method to split data in train and test samples
    :param matrix: sparse matrix of users and items
    :return: (train data, test data)
    """
    # get key matrix data: indices and values
    row, col, data = matrix.row, matrix.col, matrix.data

    # shuffle indices and split into two samples (train and test)
    indices = np.arange(row.shape[0])
    np.random.shuffle(indices)
    index_train = int(indices.shape[0] * 0.4)
    indices_train, indices_test = indices[:index_train], indices[index_train:]

    # get key matrix data about train and test samples
    row_train, col_train, data_train = row[indices_train], col[indices_train], data[indices_train]
    row_test, col_test, data_test = row[indices_test], col[indices_test], data[indices_test]

    return sparse.coo_matrix((data_train, (row_train, col_train)), shape=matrix.shape), \
           sparse.coo_matrix((data_test, (row_test, col_test)), shape=matrix.shape)
