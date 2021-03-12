import numpy as np
from scipy import sparse as sparse

from recommender_system.models.abstract_recommender_system import RecommenderSystem


class HierarchicalAlternatingLeastSquaresModel(RecommenderSystem):
    def train(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        pass

    def predict_ratings(self, user_index) -> np.ndarray:
        pass
