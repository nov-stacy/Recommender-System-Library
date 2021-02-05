import scipy.sparse as sparse
import pandas as pd
import numpy as np
from recommender_system.extra_functions import construct_coo_matrix_from_data


def read_data_from_csv(path: str) -> pd.DataFrame:
    """
    Method to load data from file with table view
    :param path: path to file with data
    :return: table
    """
    return pd.read_csv(path)


def write_data_to_npz(sparse_matrix: sparse.coo_matrix, path: str) -> None:
    """
    Method to save sparse matrix in file
    :param sparse_matrix: sparse data matrix
    :param path: path to file
    """
    sparse.save_npz(path, sparse_matrix)


def generate_sparse_matrix(table: pd.DataFrame, column_user_id, column_item_id, column_rating) -> sparse.coo_matrix:
    """
    Method for generating a two-dimensional matrix user - item from table
    :param table: data to be used
    :param column_user_id: the name of the column where the users ID are stored
    :param column_item_id: the name of the column where the items ID are stored
    :param column_rating: the name of the column where the ratings are stored
    :return: sparse matrix
    """
    # unique user ID and transformer from them to indices
    user_unique_ids = table[column_user_id].unique()
    dict_from_user_ids_to_indices = dict(zip(user_unique_ids, range(len(user_unique_ids))))

    # unique item ID and transformer from them to indices
    item_unique_ids = table[column_item_id].unique()
    dict_from_item_ids_to_indices = dict(zip(item_unique_ids, range(len(item_unique_ids))))

    # shape of matrix user - item
    matrix_shape = user_unique_ids.shape[0], item_unique_ids.shape[0]

    # indices where the data and the data itself are located
    rows = np.array(table[column_user_id].apply(lambda user_id: dict_from_user_ids_to_indices[user_id]))
    columns = np.array(table[column_item_id].apply(lambda user_id: dict_from_item_ids_to_indices[user_id]))
    data = np.array(table[column_rating])

    return construct_coo_matrix_from_data(rows, columns, data, matrix_shape)


def main(data_name: str, column_user_id: str, column_item_id: str, column_rating: str) -> None:
    """

    """
    data: pd.DataFrame = read_data_from_csv(f'data/{data_name}.csv')
    matrix = generate_sparse_matrix(data, column_user_id, column_item_id, column_rating)
    write_data_to_npz(matrix, f'data/{data_name}_matrix.npz')


if __name__ == '__main__':
    main('purchases', 'user_id', 'product_id', 'event_weight')
    main('ratings', 'user_id', 'anime_id', 'rating')
