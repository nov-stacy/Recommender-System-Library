import scipy.sparse as sparse
import numpy as np
import typing as tp


def construct_coo_matrix_from_data(rows_indices: np.ndarray, columns_indices: np.ndarray,
                                   data: np.ndarray, shape: tp.Tuple[int, int]) -> sparse.coo_matrix:
    """
    Method to generate sparse matrix from data
    :param rows_indices: line indices where values are
    :param columns_indices: column indices where values are
    :param data: values
    :param shape: matrix dimension (example: (2, 3))
    :return: sparse matrix
    """

    if len(rows_indices.shape) != 1 or len(columns_indices.shape) != 1 or len(data.shape) != 1:
        raise ValueError('Indices and data need to be 1D arrays')

    if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
        raise ValueError('Shape need to have two non-negative values')

    if not(rows_indices.shape == columns_indices.shape == data.shape):
        raise ValueError('Indices and data need to have same shape')

    return sparse.coo_matrix((data, (rows_indices, columns_indices)), shape=shape)
