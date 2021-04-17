import typing as tp
import warnings

from scipy import sparse
import matplotlib.pyplot as plt

from experiments_with_library.experiments_settings import *
from recommender_system_library.extra_functions.work_with_train_data import read_matrix_from_file
from recommender_system_library.models.implicit_models import *
from recommender_system_library.models.latent_factor_models import *
from recommender_system_library.models.abstract import EmbeddingsRecommenderSystem


RESULTS_PACKAGE = '../data_result_plots/debug_experiment'


def get_debug_values(model: EmbeddingsRecommenderSystem, data: sparse.coo_matrix, epochs: int):
    model.fit(data.astype(float), epochs=epochs, is_debug=True)
    return model.debug_information.get()


def create_plot(debug_values: tp.List[float], data_name: str, model_name: str):
    plt.plot(range(len(debug_values)), debug_values)
    plt.title(model_name)
    plt.xlabel('Iteration number')
    plt.ylabel('Error functional')
    plt.savefig(f'{RESULTS_PACKAGE}/{data_name}/{model_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model: EmbeddingsRecommenderSystem, epochs: int) -> None:
    """

    """
    data = read_matrix_from_file(f'{PACKAGE_FOR_TRAIN_DATA}/{data_name}.npz')
    debug_values = get_debug_values(model, data, epochs)
    create_plot(debug_values, data_name, str(model))


def main():

    experiments = list()

    for data, dimension in zip([MATRIX_10, MATRIX_50, MATRIX_100], [5, 25, 50]):
        experiments.extend([
            (data, StochasticLatentFactorModel(dimension, 0.0001), 30),
            (data, AlternatingLeastSquaresModel(dimension), 30),
            (data, HierarchicalAlternatingLeastSquaresModel(dimension), 30),
            (data, ImplicitStochasticLatentFactorModel(dimension, 0.0001), 30),
            #(data, ImplicitAlternatingLeastSquaresModel(dimension), 30),
            (data, ImplicitHierarchicalAlternatingLeastSquaresModel(dimension), 30),
        ])

    for data_name, model, epoch in experiments:
        generate_experiment(data_name, model, epoch)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
