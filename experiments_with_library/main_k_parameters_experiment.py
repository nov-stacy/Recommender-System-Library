import typing as tp
import warnings

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from experiments_with_library.experiments_settings import *
from recommender_system_library.extra_functions.work_with_train_data import read_matrix_from_file, get_train_matrix
from recommender_system_library.metrics import *

from recommender_system_library.models.implicit_models import ImplicitStochasticLatentFactorModel as ISLF
from recommender_system_library.models.implicit_models import ImplicitAlternatingLeastSquaresModel as IALS
from recommender_system_library.models.implicit_models import ImplicitHierarchicalAlternatingLeastSquaresModel as IHALS
from recommender_system_library.models.latent_factor_models import StochasticLatentFactorModel as SLF
from recommender_system_library.models.latent_factor_models import AlternatingLeastSquaresModel as ALS
from recommender_system_library.models.latent_factor_models import HierarchicalAlternatingLeastSquaresModel as HALS
from recommender_system_library.models.latent_factor_models import SingularValueDecompositionModel as SVD
from recommender_system_library.models.memory_based_models import UserBasedModel as UB
from recommender_system_library.models.memory_based_models import ItemBasedModel as IB

from recommender_system_library.models.abstract import AbstractRecommenderSystem

RESULTS_PACKAGE = '../data_result_plots/k_parameters_experiment'


def get_metrics(data: sparse.coo_matrix, x_train: sparse.coo_matrix, model_class: AbstractRecommenderSystem.__class__,
                name_range_parameter: str, range_parameters: tp.List[float], parameters: tp.Dict[str, float],
                train_parameters: tp.Dict[str, float]) -> tp.List[float]:
    """

    """

    items_count = data.shape[1] // 5
    users_count = data.shape[0]

    result_metrics = []

    for parameter in range_parameters:
        parameters.update({name_range_parameter: parameter})
        model = model_class(**parameters).fit(x_train.astype(float), **train_parameters)

        y_pred = [
            model.predict(user_index, items_count) for user_index in np.arange(users_count)
        ]

        y_true = [
            data.getrow(user_index).toarray()[0].argsort()[::-1][:items_count]
            for user_index in np.arange(users_count)
        ]

        result_metrics.append(precision_k(y_true, y_pred))

    return result_metrics


def create_plot(result_metrics: tp.List[float], range_parameters: tp.List[float], data_name: str, model_name: str,
                parameter_name: str):
    plt.plot(range_parameters, result_metrics)
    plt.title(model_name)
    plt.xlabel(parameter_name)
    plt.ylabel(PRECISION)
    plt.savefig(f'{RESULTS_PACKAGE}/{data_name}/{model_name}_{parameter_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model_class: AbstractRecommenderSystem.__class__,
                        name_range_parameter: str, range_parameters: tp.List[float],
                        parameters: tp.Dict[str, float], train_parameters: tp.Dict[str, float],
                        model_name: str):

    data = read_matrix_from_file(f'{PACKAGE_FOR_TRAIN_DATA}/{data_name}.npz')
    x_train = get_train_matrix(data, 0.3)

    result_metrics = get_metrics(data, x_train, model_class, name_range_parameter, range_parameters,
                                 parameters, train_parameters)

    create_plot(result_metrics, range_parameters, data_name, model_name, name_range_parameter)


def main():

    experiments = list()
    data_list = [MATRIX_10, MATRIX_50, MATRIX_100]

    for data, params in zip(data_list, [PARAMS_KNN_10, PARAMS_KNN_50, PARAMS_KNN_100]):
        experiments.extend([
            (data, UB, 'k_nearest_neighbours', params, {}, {}, 'UserBased'),
            (data, IB, 'k_nearest_neighbours', params, {}, {}, 'ItemBased'),
        ])

    for data, params in zip(data_list, [PARAMS_DIMENSION_10, PARAMS_DIMENSION_50, PARAMS_DIMENSION_100]):
        experiments.extend([
            (data, SLF, 'dimension', params, {'learning_rate': 0.0001}, {'epochs': 30}, 'SGD'),
            # (data, ALS, 'dimension', params, {}, {'epochs': 30}, 'ALS'),
            (data, HALS, 'dimension', params, {}, {'epochs': 30}, 'HALS'),
            (data, ISLF, 'dimension', params, {'learning_rate': 0.0001}, {'epochs': 30}, 'iSGD'),
            # (data, IALS, 'dimension', params, {}, {'epochs': 30}, 'iALS'),
            (data, IHALS, 'dimension', params, {}, {'epochs': 30}, 'iHALS'),
            (data, SVD, 'dimension', params, {}, {}, 'SVD')
        ])

    for data, params, dimension in zip(data_list, [PARAMS_LEARNING_RATE] * 3, [5, 25, 50]):
        experiments.extend([
            (data, SLF, 'learning_rate', params, {'dimension': dimension}, {'epochs': 30}, 'SGD'),
            (data, ISLF, 'learning_rate', params, {'dimension': dimension}, {'epochs': 30}, 'iSGD')
        ])

    for data, params, dimension in zip(data_list, [PARAMS_USER_REG] * 3, [5, 25, 50]):
        experiments.extend([
            (data, SLF, 'user_regularization', params, {'dimension': dimension, 'learning_rate': 0.0001}, {'epochs': 30},'SGD'),
            (data, ISLF, 'user_regularization', params, {'dimension': dimension, 'learning_rate': 0.0001}, {'epochs': 30}, 'iSGD')
        ])

    for data, params, dimension in zip(data_list, [PARAMS_ITEM_REG] * 3, [5, 25, 50]):
        experiments.extend([
            (data, SLF, 'item_regularization', params, {'dimension': dimension, 'learning_rate': 0.0001}, {'epochs': 30}, 'SGD'),
            (data, ISLF, 'item_regularization', params, {'dimension': dimension, 'learning_rate': 0.0001}, {'epochs': 30}, 'iSGD')
        ])

    for data, params, dimension in zip(data_list, [PARAMS_INFLUENCE_REG] * 3, [5, 25, 50]):
        experiments.extend([
            (data, ISLF, 'influence_regularization', params, {'dimension': dimension, 'learning_rate': 0.0001}, {'epochs': 30}, 'iSGD'),
            # (data, IALS, 'influence_regularization', params, {'dimension': dimension}, {'epochs': 30}, 'iALS'),
            (data, IHALS, 'influence_regularization', params, {'dimension': dimension}, {'epochs': 30}, 'iHALS')
        ])

    for experiment_parameters in experiments:
        generate_experiment(*experiment_parameters)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
