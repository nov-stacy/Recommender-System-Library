import typing as tp
import warnings

from scipy import sparse
import matplotlib.pyplot as plt

from recommender_system.extra_functions.work_with_train_data import read_data_from_npz_file
from recommender_system.models.implicit_models import *
from recommender_system.models.latent_factor_models import *
from recommender_system.models.abstract import EmbeddingsRecommenderSystem


PACKAGE = 'result_plot/debug_experiment'

MATRIX_10 = 'random_matrix_10'
MATRIX_50 = 'random_matrix_50'
MATRIX_100 = 'random_matrix_100'


def get_debug_values(model: EmbeddingsRecommenderSystem, data: sparse.coo_matrix, epochs: int):
    model.train(data.astype(float), epochs=epochs, is_debug=True)
    return model.debug_information.get()


def create_plot(debug_values: tp.List[float], data_name: str, model_name: str):
    plt.plot(range(len(debug_values)), debug_values)
    plt.title(model_name)
    plt.xlabel('Iteration number')
    plt.ylabel('Error functional')
    plt.savefig(f'{PACKAGE}/{data_name}/{model_name}.png', bbox_inches='tight')
    plt.clf()


def generate_experiment(data_name: str, model: EmbeddingsRecommenderSystem, epochs: int) -> None:
    """

    """
    data = read_data_from_npz_file(f'data/matrices/{data_name}.npz')
    debug_values = get_debug_values(model, data, epochs)
    create_plot(debug_values, data_name, str(model))


def main():

    experiments = [
        (MATRIX_10, StochasticLatentFactorModel(5, 0.0001), 30),
        (MATRIX_50, StochasticLatentFactorModel(25, 0.0001), 30),
        (MATRIX_100, StochasticLatentFactorModel(50, 0.0001), 30),

        (MATRIX_10, AlternatingLeastSquaresModel(5), 30),
        (MATRIX_50, AlternatingLeastSquaresModel(25), 30),
        (MATRIX_100, AlternatingLeastSquaresModel(50), 30),

        (MATRIX_10, HierarchicalAlternatingLeastSquaresModel(5), 30),
        (MATRIX_50, HierarchicalAlternatingLeastSquaresModel(25), 30),
        (MATRIX_100, HierarchicalAlternatingLeastSquaresModel(50), 30),

        (MATRIX_10, StochasticImplicitLatentFactorModel(5, 0.0001), 30),
        (MATRIX_50, StochasticImplicitLatentFactorModel(25, 0.0001), 30),
        (MATRIX_100, StochasticImplicitLatentFactorModel(50, 0.0001), 30),
    ]

    for data_name, model, epoch in experiments:
        generate_experiment(data_name, model, epoch)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
