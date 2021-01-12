import numpy as np
from recommender_system.abstract import RecommenderSystem


class AlternatingLeastSquares(RecommenderSystem):

    def train(self, data: np.array):
        pass

    def retrain(self, data: np.array):
        pass

    def issue_ranked_list(self, user_index: int, k_items: int):
        pass
