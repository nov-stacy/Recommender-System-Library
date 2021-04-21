import typing as tp
import warnings

from scipy import sparse
import matplotlib.pyplot as plt

from experiments_with_library.experiments_settings import *
from recommender_system_library.extra_functions.work_with_train_data import read_matrix_from_file, get_train_matrix
from recommender_system_library.models.implicit_models import *
from recommender_system_library.models.latent_factor_models import *
from recommender_system_library.models.abstract import EmbeddingsRecommenderSystem


RESULTS_PACKAGE = '../data_result_plots/debug_experiment'


def get_debug_values(model: EmbeddingsRecommenderSystem, data: sparse.coo_matrix, epochs: int, debug_name: str):
    model.fit(data.astype(float), epochs=epochs, debug_name=debug_name)
    return model.debug_information.get()


def create_plot(debug_values: tp.List[float], data_name: str, model_name: str, debug_name: str):
    plt.plot(range(len(debug_values)), debug_values)
    plt.title(model_name)
    plt.xlabel('Iteration number')
    plt.ylabel('Error functional')
    plt.savefig(f'{RESULTS_PACKAGE}/{data_name}/{debug_name}/{model_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model: EmbeddingsRecommenderSystem, epochs: int, debug_name: str) -> None:
    """

    """
    data = read_matrix_from_file(f'{PACKAGE_FOR_TRAIN_DATA}/{data_name}.npz')
    x_train = get_train_matrix(data, 0.4)
    debug_values = get_debug_values(model, x_train, epochs, debug_name)
    create_plot(debug_values, data_name, str(model), debug_name)


def main():

    experiments = list()

    for data, dimension in zip([MATRIX_10, MATRIX_50, MATRIX_100], [5, 25, 50]):
        for debug_name in DEBUG_NAMES:
            experiments.extend([
                (data, StochasticLatentFactorModel(dimension, 0.0001), dimension * 2, debug_name),
                (data, AlternatingLeastSquaresModel(dimension), dimension * 2,debug_name),
                (data, HierarchicalAlternatingLeastSquaresModel(dimension), dimension * 2, debug_name),
                (data, ImplicitStochasticLatentFactorModel(dimension, 0.0001), dimension * 2, debug_name),
                (data, ImplicitAlternatingLeastSquaresModel(dimension), dimension * 2, debug_name),
                (data, ImplicitHierarchicalAlternatingLeastSquaresModel(dimension), dimension * 2, debug_name),
            ])

    for data_name, model, epoch, debug_name in experiments:
        generate_experiment(data_name, model, epoch, debug_name)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
