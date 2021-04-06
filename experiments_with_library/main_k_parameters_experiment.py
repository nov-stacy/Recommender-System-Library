import typing as tp
import warnings

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from recommender_system_library.extra_functions.work_with_train_data import read_data_from_npz_file, get_train_data
from recommender_system_library.metrics import *
from recommender_system_library.models.implicit_models import *
from recommender_system_library.models.latent_factor_models import *
from recommender_system_library.models.memory_based_models import *
from recommender_system_library.models.abstract import AbstractRecommenderSystem

PACKAGE = 'result_plot/k_parameters_experiment'

MATRIX_10 = 'random_matrix_10'
MATRIX_50 = 'random_matrix_50'
MATRIX_100 = 'random_matrix_100'


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
        model = model_class(**parameters).train(x_train.astype(float), **train_parameters)

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
    plt.ylabel('precision@k')
    plt.savefig(f'{PACKAGE}/{data_name}/{model_name}_{parameter_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model_class: AbstractRecommenderSystem.__class__,
                        name_range_parameter: str, range_parameters: tp.List[float],
                        parameters: tp.Dict[str, float], train_parameters: tp.Dict[str, float],
                        model_name: str):

    data = read_data_from_npz_file(f'data/matrices/{data_name}.npz')
    x_train = get_train_data(data, 0.3)

    result_metrics = get_metrics(data, x_train, model_class, name_range_parameter, range_parameters,
                                 parameters, train_parameters)
    create_plot(result_metrics, range_parameters, data_name, model_name, name_range_parameter)


def main():
    params_10 = [1, 3, 5, 7]
    params_50 = [10, 20, 30, 40]
    params_100 = [20, 40, 60, 80]
    learning_rate = {'learning_rate': 0.0001}
    epochs = {'epochs': 30}

    experiments = [

        (MATRIX_10, UserBasedModel, 'k_nearest_neighbours', params_10, {}, {}, 'UserBased'),
        (MATRIX_50, UserBasedModel, 'k_nearest_neighbours', params_50, {}, {}, 'UserBased'),
        (MATRIX_100, UserBasedModel, 'k_nearest_neighbours', params_100, {}, {}, 'UserBased'),

        (MATRIX_10, ItemBasedModel, 'k_nearest_neighbours', params_10, {}, {}, 'ItemBased'),
        (MATRIX_50, ItemBasedModel, 'k_nearest_neighbours', params_50, {}, {}, 'ItemBased'),
        (MATRIX_100, ItemBasedModel, 'k_nearest_neighbours', params_100, {}, {}, 'ItemBased'),

        (MATRIX_10, AlternatingLeastSquaresModel, 'dimension', params_10, {}, epochs, 'ALS'),
        (MATRIX_50, AlternatingLeastSquaresModel, 'dimension', params_50, {}, epochs, 'ALS'),
        (MATRIX_100, AlternatingLeastSquaresModel, 'dimension', params_100, {}, epochs, 'ALS'),

        (MATRIX_10, StochasticLatentFactorModel, 'dimension', params_10, learning_rate, epochs, 'SGD'),
        (MATRIX_50, StochasticLatentFactorModel, 'dimension', params_50, learning_rate, epochs, 'SGD'),
        (MATRIX_100, StochasticLatentFactorModel, 'dimension', params_100, learning_rate, epochs, 'SGD'),

        (MATRIX_10, HierarchicalAlternatingLeastSquaresModel, 'dimension', params_10, {}, epochs, 'HALS'),
        (MATRIX_50, HierarchicalAlternatingLeastSquaresModel, 'dimension', params_50, {}, epochs, 'HALS'),
        (MATRIX_100, HierarchicalAlternatingLeastSquaresModel, 'dimension', params_100, {}, epochs, 'HALS'),

        (MATRIX_10, StochasticImplicitLatentFactorModel, 'dimension', params_10, learning_rate, epochs, 'iSGD'),
        (MATRIX_50, StochasticImplicitLatentFactorModel, 'dimension', params_50, learning_rate, epochs, 'iSGD'),
        (MATRIX_100, StochasticImplicitLatentFactorModel, 'dimension', params_100, learning_rate, epochs, 'iSGD')
    ]

    for experiment_parameters in experiments:
        generate_experiment(*experiment_parameters)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
