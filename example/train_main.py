import warnings
import numpy as np
import scipy.sparse as sparse
import typing as tp
import matplotlib.pyplot as plt
from recommender_system import models, metrics, extra_functions
from recommender_system.models.abstract_recommender_system import RecommenderSystem

USER_COUNT_PREDICT = 20
PRECISION = 'Precision@k'
RECALL = 'Recall@k'


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


class ExperimentModel:

    def __init__(self, model_class: RecommenderSystem.__class__,
                 init_params: tp.Dict[str, tp.Any], train_params: tp.Dict[str, tp.Any],
                 data_train: sparse.coo_matrix, data_test: sparse.coo_matrix) -> None:
        self.model_class: RecommenderSystem.__class__ = model_class
        self.init_params = init_params
        self.train_params = train_params
        self.data_train = data_train
        self.data_test = data_test

    def __str__(self) -> str:
        result: str = f'[[{self.model_class.__name__}]]:\n'

        if self.init_params:
            result += '[init_params]:\n' + ''.join([f'{name}: {self.init_params[name]}\n'
                                                    for name in self.init_params])

        if self.train_params:
            result += '[train_params]:\n' + ''.join([f'{name}: {self.train_params[name]}\n'
                                                     for name in self.train_params])

        return result


def generate_one_experiment(experiment_model: ExperimentModel) -> tp.Dict[str, float]:
    print(0)
    model = experiment_model.model_class(**experiment_model.init_params)
    model.train(experiment_model.data_train.astype(float), **experiment_model.train_params)
    print(1)

    predict_ratings = [
        model.predict(user_index) for user_index in np.arange(USER_COUNT_PREDICT)
    ]

    print(2)

    indices = calculate_issue_ranked_lists_for_users(predict_ratings,
                                                     experiment_model.data_test.toarray()[:USER_COUNT_PREDICT])

    print(3)

    return {
        PRECISION: metrics.precision_k(indices[0], indices[1]),
        RECALL: metrics.recall_k(indices[0], indices[1])
    }


def generate_metrics_plot(data_name: str, model_class: str,
                          param_name: str, params_list: tp.List[tp.Any],
                          experiment_models: tp.List[ExperimentModel]) -> None:

    precisions, recalls = list(), list()

    for experiment_model in experiment_models:
        experiment_result = generate_one_experiment(experiment_model)
        precisions.append(experiment_result[PRECISION])
        recalls.append(experiment_result[RECALL])

    for metric_name, metrics_list in zip([PRECISION, RECALL], [precisions, recalls]):
        plt.plot(params_list, metrics_list)
        plt.title(model_class)
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.savefig(f'result_plot/{data_name}_{model_class}_{param_name}_{metric_name}.png')
        plt.clf()


def generate_experiment(model_class: RecommenderSystem.__class__,
                        parameter_name: str, params_list: tp.List[tp.Any],
                        init_params: tp.Dict[str, tp.Any], train_params: tp.Dict[str, tp.Any],
                        data_train: sparse.coo_matrix, data_test: sparse.coo_matrix, data_name: str) -> None:

    nearest_neigbors_experiment_models = [
        ExperimentModel(
            model_class, {parameter_name: parameter, **init_params}, train_params,
            data_train, data_test
        )
        for parameter in params_list
    ]

    generate_metrics_plot(data_name, model_class.__name__, parameter_name, params_list,
                          nearest_neigbors_experiment_models)


def main_experiment():
    data_name = 'ratings'

    data = read_data_from_npz_file(f'data/{data_name}_matrix.npz')
    data_train, data_test = extra_functions.train_test_split(data)

    # generate_experiment(data_name, models.simple.NearestNeigborsModel, 'k_nearest_neigbors', [2, 4, 6, 8, 10], {}, {},
    #                     data_train, data_test)

    # generate_experiment(data_name, models.factorizing_machines.SingularValueDecompositionModel,
    #                     'dimension', [50, 100, 150], {}, {}, data_train, data_test)

    # generate_experiment(data_name, models.factorizing_machines.AlternatingLeastSquaresModel,
    #                     'dimension', [500, 1000, 1500, 2000], {'learning_rate': 0.1}, {}, data_train, data_test)


def main_train_alternating_least_squares_model():
    data_name = 'random'
    data = read_data_from_npz_file(f'data/{data_name}_matrix.npz')

    model = models.factorizing_machines.AlternatingLeastSquaresModel(dimension=75, learning_rate=0.02)
    model.train(data, iteration_number=500, debug=True)

    debug_values = model.get_debug_information()
    print(len(debug_values))

    plt.plot(range(len(debug_values)), debug_values)
    plt.title('Debug Alternating Least Squares Model')
    plt.xlabel('iteration number')
    plt.ylabel('metric')
    plt.savefig(f'result_plot/{data_name}_debug_als')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main_train_alternating_least_squares_model()
