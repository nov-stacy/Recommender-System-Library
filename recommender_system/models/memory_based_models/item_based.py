import typing as tp

import numpy as np
from scipy import sparse as sparse
from scipy.spatial.distance import correlation
from sklearn.neighbors import NearestNeighbors

from recommender_system.models.abstract_recommender_system import RecommenderSystem


class ItemBasedCollaborativeFilteringModel(RecommenderSystem):

    def __init__(self, k_nearest_neigbors: int) -> None:
        """
        Parameters
        ----------
        k_nearest_neigbors:
            The number of closest neighbors that must be taken into account
            when predicting an estimate for an item
        """
        self.__data: sparse.coo_matrix = np.array([])  # matrix for storing all data
        self.__mean_items: np.ndarray = np.array([])  # matrix for the average ratings of each user
        self.__mean_users: np.ndarray = np.array([])  # matrix for the average ratings of each item

        # algorithm for determining the nearest neighbors
        self.__knn: NearestNeighbors = NearestNeighbors(n_neighbors=k_nearest_neigbors + 1, metric=correlation)  # TODO

    def train(self, data: sparse.coo_matrix) -> 'RecommenderSystem':
        self.__data: sparse.coo_matrix = data
        self.__mean_items: np.ndarray = self.__data.mean(axis=0).transpose()
        self.__mean_users: np.ndarray = self.__data.mean(axis=1)
        self.__knn.fit(self.__data.transpose())
        return self

    def predict_ratings(self, user_index) -> np.ndarray:
        raise NotImplementedError('There is no such method in this class')

    def predict(self, user_index, items_count: int) -> np.ndarray:
        # getting the indices of all items that the user has viewed
        items: np.ndarray = np.where(self.__data.getrow(user_index).toarray()[0] != 0)[0]

        # get a list of k nearest neigbors of items
        nearest_items: tp.Dict[int, float] = dict()

        for item_index in items:

            # get a list of k nearest neigbors of current item
            distances, items = self.__knn.kneighbors(self.__data.getcol(item_index), return_distance=True)
            distances = distances[items != item_index]
            items = items[items != item_index]

            # update dictionary of distances for each item
            for item, distance in zip(items, distances):
                if item not in nearest_items:
                    nearest_items[item] = distance
                else:
                    nearest_items[item] = min(nearest_items[item], distance)

        # get items for recommendation
        sorted_items = sorted(nearest_items.items(), key=lambda x: x[1])[:items_count]
        return np.array(list(zip(*sorted_items))[0])
