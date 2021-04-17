import typing as tp
import warnings

from scipy import sparse
import matplotlib.pyplot as plt

from experiments_with_library.experiments_settings import *
from recommender_system_library.extra_functions.work_with_predict_data import calculate_predicted_items
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

from recommender_system_library.models.abstract import AbstractRecommenderSystem


RESULTS_PACKAGE = '../data_result_plots/ratings_parameters_experiment'


def get_indices(data: sparse.coo_matrix, model: AbstractRecommenderSystem, users_count: int):

    true_indices, predicted_indices = list(), list()

    for user_index in range(users_count):

        predicted_ratings = model.predict_ratings(user_index)
        true_ratings = data.getrow(user_index).toarray()[0]

        barrier_value = true_ratings.mean()
        true_indices.append(calculate_predicted_items(predicted_ratings, barrier_value=barrier_value))
        predicted_indices.append(calculate_predicted_items(true_ratings, barrier_value=barrier_value))

    return true_indices, predicted_indices


def get_metrics(data: sparse.coo_matrix, x_train: sparse.coo_matrix, model_class: AbstractRecommenderSystem.__class__,
                name_range_parameter: str, range_parameters: tp.List[float], parameters: tp.Dict[str, float],
                train_parameters: tp.Dict[str, float]) -> tp.Dict[str, tp.List[float]]:
    """

    """

    users_count = data.shape[0]

    result_metrics = {
        PRECISION: list(),
        RECALL: list()
    }

    for parameter in range_parameters:
        parameters.update({name_range_parameter: parameter})
        model = model_class(**parameters).fit(x_train.astype(float), **train_parameters)

        true_indices, predicted_indices = get_indices(data, model, users_count)

        result_metrics[PRECISION].append(precision_k(true_indices, predicted_indices))
        result_metrics[RECALL].append(recall_k(true_indices, predicted_indices))

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
                        model_name: str):

    data = read_matrix_from_file(f'{PACKAGE_FOR_TRAIN_DATA}/{data_name}.npz')
    x_train = get_train_matrix(data, 0.2)

    result_metrics = get_metrics(data, x_train, model_class, name_range_parameter, range_parameters,
                                 parameters, train_parameters)

    for key in result_metrics:
        create_plot(result_metrics[key], range_parameters, data_name, model_name, name_range_parameter, key)


def main():

    experiments = list()
    data_list = [MATRIX_10, MATRIX_50, MATRIX_100]

    for data, params in zip(data_list, [PARAMS_KNN_10, PARAMS_KNN_50, PARAMS_KNN_100]):
        experiments.extend([
            (data, UB, 'k_nearest_neighbours', params, {}, {}, 'UserBased')
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
            (data, SLF, 'user_regularization', params, {'dimension': dimension, 'learning_rate': 0.0001}, {'epochs': 30}, 'SGD'),
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
