import typing as tp
import warnings

from scipy import sparse
import matplotlib.pyplot as plt

from recommender_system.extra_functions.work_with_predict_data import calculate_issue_ranked_list
from recommender_system.extra_functions.work_with_train_data import read_data_from_npz_file, get_train_data
from recommender_system.metrics import *
from recommender_system.models.implicit_models import *
from recommender_system.models.latent_factor_models import *
from recommender_system.models.memory_based_models import *
from recommender_system.models.abstract import AbstractRecommenderSystem


PACKAGE = 'result_plot/ratings_parameters_experiment'

PRECISION = 'precision@k'
RECALL = 'recall@k'

MATRIX_10 = 'random_matrix_10'
MATRIX_50 = 'random_matrix_50'
MATRIX_100 = 'random_matrix_100'


def get_indices(data: sparse.coo_matrix, model: AbstractRecommenderSystem, users_count: int):

    true_indices, predicted_indices = list(), list()

    for user_index in range(users_count):

        predicted_ratings = model.predict_ratings(user_index)
        true_ratings = data.getrow(user_index).toarray()[0]

        barrier_value = predicted_ratings.mean()
        true_indices.append(calculate_issue_ranked_list(predicted_ratings, barrier_value=barrier_value))
        predicted_indices.append(calculate_issue_ranked_list(true_ratings, barrier_value=barrier_value))

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
        model = model_class(**parameters).train(x_train.astype(float), **train_parameters)

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
    plt.savefig(f'{PACKAGE}/{data_name}/{metric_name}/{model_name}_{parameter_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model_class: AbstractRecommenderSystem.__class__,
                        name_range_parameter: str, range_parameters: tp.List[float],
                        parameters: tp.Dict[str, float], train_parameters: tp.Dict[str, float],
                        model_name: str):

    data = read_data_from_npz_file(f'data/matrices/{data_name}.npz')
    x_train = get_train_data(data, 0.2)
    result_metrics = get_metrics(data, x_train, model_class, name_range_parameter, range_parameters,
                                 parameters, train_parameters)
    for key in result_metrics:
        create_plot(result_metrics[key], range_parameters, data_name, model_name, name_range_parameter, key)


def main():

    params_10 = [1, 3, 5, 7]
    params_50 = [10, 20, 30, 40]
    params_100 = [20, 40, 60, 80]
    learning_rate = {'learning_rate': 0.0001}
    epochs = {'epochs': 30}

    experiments = [
        (MATRIX_10, UserBasedModel, 'k_nearest_neigbors', params_10, {}, {}, 'UserBased'),
        (MATRIX_50, UserBasedModel, 'k_nearest_neigbors', params_50, {}, {}, 'UserBased'),
        (MATRIX_100, UserBasedModel, 'k_nearest_neigbors', params_100, {}, {}, 'UserBased'),
        
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
