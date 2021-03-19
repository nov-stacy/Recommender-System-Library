import warnings

import pandas as pd

from recommender_system.extra_functions import work_with_csv_data


def main(data_name: str, column_user_id: str, column_item_id: str, column_rating: str) -> None:
    """

    """
    data: pd.DataFrame = work_with_csv_data.read_data_from_csv(f'data/tables/{data_name}.csv')
    matrix = work_with_csv_data.generate_sparse_matrix(data, column_user_id, column_item_id, column_rating)
    work_with_csv_data.write_data_to_npz(matrix, f'data/matrices/{data_name}_matrix.npz')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main('purchases', 'user_id', 'product_id', 'event_weight')
    main('ratings', 'user_id', 'anime_id', 'rating')
