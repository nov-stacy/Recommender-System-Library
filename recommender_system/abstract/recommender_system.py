from abc import ABC, abstractmethod
import numpy as np


class RecommenderSystem(ABC):

    @abstractmethod
    def train(self, data: np.array) -> 'RecommenderSystem':
        pass

    @abstractmethod
    def retrain(self, data: np.array) -> 'RecommenderSystem':
        pass

    @abstractmethod
    def issue_ranked_list(self, user_index: int, k_items: int) -> np.array:
        pass
