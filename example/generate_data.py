import scipy.sparse as sparse
import typing as tp
import numpy as np


def read_data(path: str) -> sparse.spmatrix:
    """
    Method to load data in sparse view
    :param path: path to file with data
    :return: sparse matrix
    """
    return sparse.load_npz(path)


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

    return sparse.coo_matrix((data_test, (row_train, col_train)), shape=matrix.shape), \
           sparse.coo_matrix((data_test, (row_test, col_test)), shape=matrix.shape)
