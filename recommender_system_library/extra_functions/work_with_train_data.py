import scipy.sparse as sparse
import numpy as np
import typing as tp


def read_data_from_npz_file(path: str) -> sparse.coo_matrix:
    """
    Method to load data from file with sparse view

    Parameters
    ----------
    path: str
        Path to file with data

    Returns
    -------
    matrix: sparse matrix
    """

    return sparse.load_npz(path)


def construct_coo_matrix_from_data(rows_indices: np.ndarray, columns_indices: np.ndarray,
                                   data: np.ndarray, shape: tp.Tuple[int, int]) -> sparse.coo_matrix:
    """
    Method to generate sparse matrix from data

    Parameters
    ----------
    rows_indices: numpy array
        Line indices where values are
    columns_indices: numpy array
        Column indices where values are
    data: numpy array
        Values
    shape: (int, int)
        Matrix dimension (example: (2, 3))

    Returns
    -------
    matrix: sparse matrix
    """

    if len(rows_indices.shape) != 1 or len(columns_indices.shape) != 1 or len(data.shape) != 1:
        raise ValueError('Indices and data need to be 1D arrays')

    if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
        raise ValueError('Shape need to have two non-negative values')

    if not(rows_indices.shape == columns_indices.shape == data.shape):
        raise ValueError('Indices and data need to have same shape')

    return sparse.coo_matrix((data, (rows_indices, columns_indices)), shape=shape)


def train_test_split(matrix: sparse.coo_matrix, proportion: float) -> tp.Tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """
    Method to split data in train and test samples

    Parameters
    ----------
    matrix: sparse matrix
        2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
    proportion: float
        The relative amount of data in the training dataset

    Returns
    -------
    (train data, test data): (sparse matrix, sparse matrix)
    """

    if proportion <= 0 or proportion >= 1:
        raise ValueError('Proportion need to be in (0, 1)')

    # get key matrix data: indices and values
    row, col, data = matrix.row, matrix.col, matrix.data

    # shuffle indices and split into two samples (train and test)
    indices = np.arange(row.shape[0])
    np.random.shuffle(indices)
    index_train = int(indices.shape[0] * proportion)
    indices_train, indices_test = indices[:index_train], indices[index_train:]

    # get key matrix data about train and test samples
    row_train, col_train, data_train = row[indices_train], col[indices_train], data[indices_train]
    row_test, col_test, data_test = row[indices_test], col[indices_test], data[indices_test]

    return sparse.coo_matrix((data_train, (row_train, col_train)), shape=matrix.shape), \
           sparse.coo_matrix((data_test, (row_test, col_test)), shape=matrix.shape)


def get_train_data(matrix: sparse.coo_matrix, proportion: float) -> sparse.coo_matrix:
    """
    Method to get train data

    Parameters
    ----------
    matrix: sparse matrix
        2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
    proportion: float
        The relative amount of data in the training dataset

    Returns
    -------
    train data: sparse matrix
    """

    if proportion <= 0 or proportion >= 1:
        raise ValueError('Proportion need to be in (0, 1)')

    # get key matrix data: indices and values
    row, col, data = matrix.row, matrix.col, matrix.data

    # shuffle indices and get a part of data
    indices = np.arange(row.shape[0])
    np.random.shuffle(indices)
    indices_train = indices[:int(indices.shape[0] * proportion)]

    # get key matrix data about train samples
    row_train, col_train, data_train = row[indices_train], col[indices_train], data[indices_train]

    return sparse.coo_matrix((data_train, (row_train, col_train)), shape=matrix.shape)