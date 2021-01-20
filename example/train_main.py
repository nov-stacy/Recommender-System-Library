import numpy as np
import scipy.sparse as sparse
import typing as tp
from recommender_system import models, metrics, extra_functions


def read_data_from_npz_file(path: str) -> sparse.spmatrix:
    """
    Method to load data from file with sparse view
    :param path: path to file with data
    :return: sparse matrix
    """
    return sparse.load_npz(path)


def calculate_issue_ranked_lists_for_users(data_predict_ratings: np.ndarray,
                                           data_test_ratings: np.ndarray) -> tp.Tuple[tp.List[np.ndarray],
                                                                                      tp.List[np.ndarray]]:
    """
    Method to calculate indices of items for ratings
    :param data_predict_ratings: ratings that were predicted
    :param data_test_ratings: ratings from test data
    :return: indices
    """
    true_indices, predicted_indices = list(), list()

    for predicted_ratings, test_ratings in zip(data_predict_ratings, data_test_ratings):
        barrier_value = predicted_ratings.mean()
        true_indices.append(extra_functions.calculate_issue_ranked_list(predicted_ratings, barrier_value=barrier_value))
        predicted_indices.append(extra_functions.calculate_issue_ranked_list(test_ratings, barrier_value=barrier_value))

    return true_indices, predicted_indices


if __name__ == '__main__':

    model = models.factorizing_machines.svd.SingularValueDecompositionNaiveModel(100)

    data = read_data_from_npz_file('data/matrix.npz')
    data_train, data_test = extra_functions.train_test_split(data)

    model.train(data_train.astype(float))
    user_count_predict = 20

    predict_ratings = [
        model.predict(user_index) for user_index in np.arange(user_count_predict)
    ]

    indices = calculate_issue_ranked_lists_for_users(predict_ratings, data_test.toarray()[:user_count_predict])

    print('Precision@k:', metrics.precision_k(indices[0], indices[1]))
    print('Recall@k:', metrics.recall_k(indices[0], indices[1]))
