import typing as tp
import warnings

from scipy import sparse
import matplotlib.pyplot as plt

from experiments_with_library.experiments_settings import *
from recommender_system_library.extra_functions.work_with_ratings import calculate_predicted_items
from recommender_system_library.extra_functions.work_with_matrices import read_matrix_from_file, get_train_matrix
from recommender_system_library.metrics import *

from recommender_system_library.models.implicit_models import ImplicitStochasticLatentFactorModel as ISLFM
from recommender_system_library.models.implicit_models import ImplicitAlternatingLeastSquaresModel as IALS
from recommender_system_library.models.implicit_models import ImplicitHierarchicalAlternatingLeastSquaresModel as IHALS
from recommender_system_library.models.latent_factor_models import StochasticLatentFactorModel as SLFM
from recommender_system_library.models.latent_factor_models import AlternatingLeastSquaresModel as ALS
from recommender_system_library.models.latent_factor_models import HierarchicalAlternatingLeastSquaresModel as HALS
from recommender_system_library.models.latent_factor_models import SingularValueDecompositionModel as SVD
from recommender_system_library.models.memory_based_models import UserBasedModel as UB

from recommender_system_library.models.abstract import AbstractRecommenderSystem

RESULTS_PACKAGE = '../data_result_plots/ratings_parameters_experiment'


def get_result_for_items(data: sparse.coo_matrix, model: AbstractRecommenderSystem, users_count: int):
    true_indices, predicted_indices = list(), list()

    for user_index in range(users_count):
        predicted_ratings = model.predict_ratings(user_index)
        true_ratings = data.getrow(user_index).toarray()[0]

        barrier_value = true_ratings.mean()
        true_indices.append(calculate_predicted_items(predicted_ratings, barrier_value=barrier_value))
        predicted_indices.append(calculate_predicted_items(true_ratings, barrier_value=barrier_value))

    return true_indices, predicted_indices


def get_result_for_ratings(data: sparse.coo_matrix, model: AbstractRecommenderSystem, users_count: int):
    true_ratings_list, predicted_ratings_list = list(), list()

    for user_index in range(users_count):
        predicted_ratings = model.predict_ratings(user_index)
        true_ratings = data.getrow(user_index).toarray()[0]

        true_ratings_list.append(true_ratings)
        predicted_ratings_list.append(predicted_ratings)

    return true_ratings_list, predicted_ratings_list


def get_result_for_interest(data: sparse.coo_matrix, model: AbstractRecommenderSystem, users_count: int):
    true_interest_list, predicted_ratings_list = list(), list()

    for user_index in range(users_count):
        predicted_ratings = model.predict_ratings(user_index)
        true_ratings = data.getrow(user_index).toarray()[0]

        barrier_value = true_ratings.mean()
        true_interest = (true_ratings >= barrier_value).astype(int)
        true_interest_list.append(true_interest)
        predicted_ratings_list.append(predicted_ratings)

    return true_interest_list, predicted_ratings_list


def get_metrics(data: sparse.coo_matrix, x_train: sparse.coo_matrix, model_class: AbstractRecommenderSystem.__class__,
                name_range_parameter: str, range_parameters: tp.List[float], parameters: tp.Dict[str, float],
                train_parameters: tp.Dict[str, float]) -> tp.Dict[str, tp.List[float]]:
    users_count = data.shape[0]

    result_metrics = {
        key: list() for key in list(METRICS_FOR_RATINGS_NAMES.keys()) + list(METRICS_FOR_ITEMS_NAMES.keys()) +
                               list(METRICS_FOR_INTEREST_NAMES.keys())
    }

    for parameter in range_parameters:
        parameters.update({name_range_parameter: parameter})
        model = model_class(**parameters).fit(x_train.astype(float), **train_parameters)

        for key in result_metrics:

            if key in METRICS_FOR_RATINGS_NAMES:
                true_result, predicted_result = get_result_for_ratings(data, model, users_count)
                result_metrics[key].append(METRICS_FOR_RATINGS_NAMES[key](true_result, predicted_result))
            elif key in METRICS_FOR_ITEMS_NAMES:
                true_result, predicted_result = get_result_for_items(data, model, users_count)
                result_metrics[key].append(METRICS_FOR_ITEMS_NAMES[key](true_result, predicted_result))
            else:
                true_result, predicted_result = get_result_for_interest(data, model, users_count)
                result_metrics[key].append(METRICS_FOR_INTEREST_NAMES[key](true_result, predicted_result))

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
    EPOCHS = {'epochs': 30}
    K = 'k_nearest_neighbours'
    USER_REG, ITEM_REG, INF_REG = 'user_regularization', 'item_regularization', 'influence_regularization'
    DIM, LR = 'dimension', 'learning_rate'
    DIM_VALUES = [5, 10, 10, 25, 30, 30, 50, 200]

    experiments = list()
    count = 3

    for data, params in zip(MATRICES[3:], [PARAMS_KNN_50, PARAMS_KNN_50, PARAMS_KNN_50]):
        experiments.extend([
            (data, UB, K, params, {}, {}, 'UserBased')
        ])

    for data, params in zip(MATRICES[3:], [PARAMS_DIMENSION_50, PARAMS_DIMENSION_50, PARAMS_DIMENSION_50]):
        experiments.extend([
            (data, SLFM, DIM, params, {LR: 0.0001}, EPOCHS, 'SGD'),
            (data, ALS, DIM, params, {}, EPOCHS, 'ALS'),
            (data, HALS, DIM, params, {}, EPOCHS, 'HALS'),
            (data, ISLFM, DIM, params, {LR: 0.0001}, EPOCHS, 'iSGD'),
            (data, IALS, DIM, params, {}, EPOCHS, 'iALS'),
            (data, IHALS, DIM, params, {}, EPOCHS, 'iHALS'),
            (data, SVD, DIM, params, {}, {}, 'SVD')
        ])

    for data, params, dimension in zip(MATRICES[3:], [PARAMS_LEARNING_RATE] * count, DIM_VALUES):
        experiments.extend([
            (data, SLFM, LR, params, {DIM: dimension}, EPOCHS, 'SGD'),
            (data, ISLFM, LR, params, {DIM: dimension}, EPOCHS, 'iSGD')
        ])

    for data, params, dimension in zip(MATRICES[3:], [PARAMS_USER_REG] * count, DIM_VALUES):
        experiments.extend([
            (data, SLFM, USER_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'SGD'),
            (data, ISLFM, USER_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'iSGD')
        ])

    for data, params, dimension in zip(MATRICES[3:], [PARAMS_ITEM_REG] * count, DIM_VALUES):
        experiments.extend([
            (data, SLFM, ITEM_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'SGD'),
            (data, ISLFM, ITEM_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'iSGD')
        ])

    for data, params, dimension in zip(MATRICES[3:], [PARAMS_INFLUENCE_REG] * count, DIM_VALUES):
        experiments.extend([
            (data, ISLFM, INF_REG, params, {DIM: dimension, LR: 0.0001}, EPOCHS, 'iSGD'),
            (data, IALS, INF_REG, params, {DIM: dimension}, EPOCHS, 'iALS'),
            (data, IHALS, INF_REG, params, {DIM: dimension}, EPOCHS, 'iHALS')
        ])

    for experiment_parameters in experiments:
        generate_experiment(*experiment_parameters)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
