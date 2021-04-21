import numpy as _np

__all__ = [
    'PACKAGE_FOR_TRAIN_DATA',
    'MATRIX_10', 'MATRIX_50', 'MATRIX_100',
    'PRECISION', 'RECALL',
    'PARAMS_KNN_10',
] + [
    f'MATRIX_{name}' for name in [10, 50, 100]
] + [
    f'PARAMS_KNN_{name}' for name in [10, 50, 100]
] + [
    f'PARAMS_DIMENSION_{name}' for name in [10, 50, 100]
] + [
    'PARAMS_LEARNING_RATE', 'PARAMS_USER_REG', 'PARAMS_ITEM_REG',
    'PARAMS_INFLUENCE_REG', 'DEBUG_NAMES'
]

PACKAGE_FOR_TRAIN_DATA = '../data_train/matrices'

MATRIX_10 = 'random_matrix_10'
MATRIX_50 = 'random_matrix_50'
MATRIX_100 = 'random_matrix_100'

PRECISION = 'precision@k'
RECALL = 'recall@k'

PARAMS_KNN_10 = _np.arange(1, 10, 2)
PARAMS_KNN_50 = _np.arange(1, 50, 2)
PARAMS_KNN_100 = _np.arange(1, 100, 2)

PARAMS_DIMENSION_10 = _np.arange(1, 10, 2)
PARAMS_DIMENSION_50 = _np.arange(1, 50, 2)
PARAMS_DIMENSION_100 = _np.arange(1, 100, 2)

PARAMS_LEARNING_RATE = PARAMS_USER_REG = PARAMS_ITEM_REG = PARAMS_INFLUENCE_REG = _np.arange(0, 1, 0.1)

DEBUG_NAMES = ['mse', 'rmse', 'mae']
