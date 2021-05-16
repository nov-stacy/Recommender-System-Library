import numpy as _np

from recommender_system_library.metrics import *


__all__ = [
    'PACKAGE_FOR_TRAIN_DATA',
    'MATRICES'
] + [
    'METRICS_FOR_RATINGS_NAMES',
    'METRICS_FOR_ITEMS_NAMES',
    'METRICS_FOR_INTEREST_NAMES',
    'ERROR_NAMES'
] + [
    f'PARAMS_KNN_{name}' for name in [10, 50, 100, 200]
] + [
    f'PARAMS_DIMENSION_{name}' for name in [10, 50, 100, 200]
] + [
    'PARAMS_LEARNING_RATE', 'PARAMS_USER_REG', 'PARAMS_ITEM_REG',
    'PARAMS_INFLUENCE_REG'
]

PACKAGE_FOR_TRAIN_DATA = '../data_train/matrices'

MATRICES = [
    'random_10_10_matrix', 'random_10_20_matrix', 'random_20_10_matrix',
    'random_50_50_matrix', 'random_50_100_matrix', 'random_100_50_matrix',
    'random_100_100_matrix',
    'ratings_matrix'
]

METRICS_FOR_RATINGS_NAMES = {
    'ndcg': normalized_discounted_cumulative_gain
}
METRICS_FOR_ITEMS_NAMES = {
    'f1': f1_measure,
    'precision@k': precision_k,
    'recall@k': recall_k
}
METRICS_FOR_INTEREST_NAMES = {
    'auc': roc_auc
}
ERROR_NAMES = {
    'mse': mean_square_error,
    'rmse': root_mean_square_error,
    'mae': mean_absolute_error
}

PARAMS_KNN_10 = _np.arange(1, 10, 2)
PARAMS_KNN_50 = _np.arange(1, 50, 2)
PARAMS_KNN_100 = _np.arange(1, 100, 2)
PARAMS_KNN_200 = _np.arange(1, 202, 4)

PARAMS_DIMENSION_10 = _np.arange(3, 10, 2)
PARAMS_DIMENSION_50 = _np.arange(1, 50, 2)
PARAMS_DIMENSION_100 = _np.arange(1, 100, 2)
PARAMS_DIMENSION_200 = _np.arange(1, 202, 4)

PARAMS_LEARNING_RATE = PARAMS_USER_REG = PARAMS_ITEM_REG = PARAMS_INFLUENCE_REG = _np.arange(0, 1, 0.1)
