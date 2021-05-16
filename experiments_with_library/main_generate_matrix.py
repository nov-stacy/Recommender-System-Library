import warnings
import os

import pandas as pd

from recommender_system_library.extra_functions import work_with_tables, work_with_matrices


PATH_TO_TABLES = '../data_train/tables'
PATH_TO_MATRICES = '../data_train/matrices'


def main(data_name: str, column_user_id: str, column_item_id: str, column_rating: str) -> None:
    data: pd.DataFrame = work_with_tables.read_data_from_csv(f'{PATH_TO_TABLES}/{data_name}.csv')
    matrix = work_with_tables.generate_sparse_matrix(data, column_user_id, column_item_id, column_rating)
    work_with_matrices.write_matrix_to_file(matrix, f'{PATH_TO_MATRICES}/{data_name}_matrix.npz')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    for table in os.listdir(PATH_TO_TABLES):
        main(table[:-4], 'user_id', 'item_id', 'rating')

