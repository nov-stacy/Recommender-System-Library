import numpy as np
from scipy import sparse as sparse

from recommender_system.models.abstract import RecommenderSystem


class ImplicitHierarchicalAlternatingLeastSquaresModel(RecommenderSystem):

    def train(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        pass

    def predict_ratings(self, user_index) -> np.ndarray:
        pass

# TODO

