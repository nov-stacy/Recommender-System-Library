from abc import ABC, abstractmethod


class RecommenderSystem(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def retrain(self):
        pass

    @abstractmethod
    def issue_ranked_list(self):
        pass
