import typing as tp
import warnings

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from experiments_with_library.experiments_settings import *
from recommender_systems.extra_functions.work_with_matrices import read_matrix_from_file, get_train_matrix

from recommender_systems.models.implicit_models import ImplicitStochasticLatentFactorModel as ISLFM
from recommender_systems.models.implicit_models import ImplicitAlternatingLeastSquaresModel as IALS
from recommender_systems.models.implicit_models import ImplicitHierarchicalAlternatingLeastSquaresModel as IHALS
from recommender_systems.models.latent_factor_models import StochasticLatentFactorModel as SLFM
from recommender_systems.models.latent_factor_models import AlternatingLeastSquaresModel as ALS
from recommender_systems.models.latent_factor_models import HierarchicalAlternatingLeastSquaresModel as HALS
from recommender_systems.models.latent_factor_models import SingularValueDecompositionModel as SVD
from recommender_systems.models.memory_based_models import UserBasedModel as UB
from recommender_systems.models.memory_based_models import ItemBasedModel as IB

from recommender_systems.models.abstract import AbstractRecommenderSystem

RESULTS_PACKAGE = '../data_result_plots/k_parameters_experiment'


def get_metrics(data: sparse.coo_matrix, x_train: sparse.coo_matrix, model_class: AbstractRecommenderSystem.__class__,
                name_range_parameter: str, range_parameters: tp.List[float], parameters: tp.Dict[str, float],
                train_parameters: tp.Dict[str, float], metric_name: str) -> tp.List[float]:

    items_count = data.shape[1] // 5
    users_count = data.shape[0]

    result_metrics = []

    for parameter in range_parameters:
        parameters.update({name_range_parameter: parameter})
        model = model_class(**parameters).fit(x_train.astype(float), **train_parameters)

        y_pred = [
            model.predict(user_index)[:items_count] for user_index in np.arange(users_count)
        ]

        y_true = [
            data.getrow(user_index).toarray()[0].argsort()[::-1][:items_count]
            for user_index in np.arange(users_count)
        ]

        result_metrics.append(METRICS_FOR_ITEMS_NAMES[metric_name](y_true, y_pred))

    return result_metrics


def create_plot(result_metrics: tp.List[float], range_parameters: tp.List[float], data_name: str, model_name: str,
                parameter_name: str, metric_name: str):
    plt.plot(range_parameters, result_metrics)
    plt.title(model_name)
    plt.xlabel(parameter_name)
    plt.ylabel(metric_name)
    plt.savefig(f'{RESULTS_PACKAGE}/{data_name}/{metric_name}/{model_name}_{parameter_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model_class: AbstractRecommenderSystem.__class__,
                        name_range_parameter: str, range_parameters: tp.List[float],
                        parameters: tp.Dict[str, float], train_parameters: tp.Dict[str, float],
                        model_name: str, metric_name: str):

    data = read_matrix_from_file(f'{PACKAGE_FOR_TRAIN_DATA}/{data_name}.npz')
    x_train = get_train_matrix(data, 0.3)

    result_metrics = get_metrics(data, x_train, model_class, name_range_parameter, range_parameters,
                                 parameters, train_parameters, metric_name)

    create_plot(result_metrics, range_parameters, data_name, model_name, name_range_parameter, metric_name)


def main():

    EPOCHS = {'epochs': 30}
    K = 'k_nearest_neighbours'
    USER_REG, ITEM_REG, INF_REG = 'user_regularization', 'item_regularization', 'influence_regularization'
    DIM, LR = 'dimension', 'learning_rate'
    DIM_VALUES = [5, 10, 10, 25, 30, 30, 50, 200]

    experiments = list()
    count = 3

    for metric_name in METRICS_FOR_ITEMS_NAMES:

        for data, params in zip(MATRICES, [PARAMS_KNN_10, PARAMS_KNN_10, PARAMS_KNN_10]):
            experiments.extend([
                (data, UB, K, params, {}, {}, 'UserBased', metric_name),
                (data, IB, K, params, {}, {}, 'ItemBased', metric_name),
            ])

        for data, params in zip(MATRICES, [PARAMS_DIMENSION_10, PARAMS_DIMENSION_10, PARAMS_DIMENSION_10]):
            experiments.extend([
                (data, SLFM, DIM, params, {LR: 0.0001}, EPOCHS, 'SGD', metric_name),
                (data, ALS, DIM, params, {}, EPOCHS, 'ALS', metric_name),
                (data, HALS, DIM, params, {}, EPOCHS, 'HALS', metric_name),
                (data, ISLFM, DIM, params, {LR: 0.0001}, EPOCHS, 'iSGD', metric_name),
                (data, IALS, DIM, params, {}, EPOCHS, 'iALS', metric_name),
                (data, IHALS, DIM, params, {}, EPOCHS, 'iHALS', metric_name),
                (data, SVD, DIM, params, {}, {}, 'SVD', metric_name)
            ])

        for data, params, dimension in zip(MATRICES, [PARAMS_LEARNING_RATE] * count, DIM_VALUES):
            experiments.extend([
                (data, SLFM, LR, params, {DIM: dimension}, EPOCHS, 'SGD', metric_name),
                (data, ISLFM, LR, params, {DIM: dimension}, EPOCHS, 'iSGD', metric_name)
            ])

        for data, params, dimension in zip(MATRICES, [PARAMS_USER_REG] * count, DIM_VALUES):
            experiments.extend([
                (data, SLFM, USER_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'SGD', metric_name),
                (data, ISLFM, USER_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'iSGD', metric_name)
            ])

        for data, params, dimension in zip(MATRICES, [PARAMS_ITEM_REG] * count, DIM_VALUES):
            experiments.extend([
                (data, SLFM, ITEM_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'SGD', metric_name),
                (data, ISLFM, ITEM_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'iSGD', metric_name)
            ])

        for data, params, dimension in zip(MATRICES, [PARAMS_INFLUENCE_REG] * count, DIM_VALUES):
            experiments.extend([
                (data, ISLFM, INF_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'iSGD', metric_name),
                (data, IALS, INF_REG, params, {DIM: dimension}, EPOCHS, 'iALS', metric_name),
                (data, IHALS, INF_REG, params, {DIM: dimension}, EPOCHS, 'iHALS', metric_name)
            ])

    for experiment_parameters in experiments:
        generate_experiment(*experiment_parameters)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
