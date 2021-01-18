import numpy as np
import typing as tp
from recommender_system.abstract import RecommenderSystem


class SingularValueDecompositionModel(RecommenderSystem):

    def __init__(self):
        pass

    def __calculate_ratings(self, user_index: int) -> tp.List[tp.Tuple[int, int]]:
        pass

    def train(self, data: np.array) -> 'SingularValueDecompositionModel':
        pass

    def retrain(self, data: np.array) -> 'SingularValueDecompositionModel':
        pass

    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        pass
