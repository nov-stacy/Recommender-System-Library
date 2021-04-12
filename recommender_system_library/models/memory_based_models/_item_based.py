import typing as tp

import numpy as np
from scipy import sparse as sparse
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from recommender_system_library.models.abstract import OneEpochAbstractRecommenderSystem


class ItemBasedModel(OneEpochAbstractRecommenderSystem):
    """
    Recommender system based on similarities between items

    Realization
    -----------
    Determine the nearest neighbors for the items, which the user marked, and get the items,
    which are closest to the original ones
    """

    @staticmethod
    def _correlation(first, second):
        return 1 - pearsonr(first.toarray()[0], second.toarray()[0])[0]

    def __init__(self, k_nearest_neighbours: int) -> None:
        """
        Parameters
        ----------
        k_nearest_neighbours:
            The number of closest neighbors that must be taken into account
            when predicting an estimate for an item
        """
        self._data: sparse.coo_matrix = np.array([])  # matrix for storing all data
        self._mean_items: np.ndarray = np.array([])  # matrix for the average ratings of each user
        self._mean_users: np.ndarray = np.array([])  # matrix for the average ratings of each item

        # algorithm for determining the nearest neighbors
        self._k_nearest_neighbours = k_nearest_neighbours
        self._knn: NearestNeighbors = NearestNeighbors(n_neighbors=k_nearest_neighbours + 1, metric=self._correlation)

    def fit(self, data: sparse.coo_matrix) -> 'OneEpochAbstractRecommenderSystem':
        self._data: sparse.coo_matrix = data
        self._mean_items: np.ndarray = self._data.mean(axis=0).transpose()
        self._mean_users: np.ndarray = self._data.mean(axis=1)
        self._knn.fit(self._data.transpose())
        self._is_trained = True
        return self

    def predict_ratings(self, user_index: int) -> np.ndarray:
        raise NotImplemented

    def predict(self, user_index, items_count: int) -> np.ndarray:

        self._is_predict()

        # getting the indices of all items that the user has viewed
        items: np.ndarray = np.where(self._data.getrow(user_index).toarray()[0] != 0)[0]

        # get a list of k nearest neighbours of items
        nearest_items: tp.Dict[int, float] = dict()

        for item_index in items:

            # get a list of k nearest neighbours of current item
            distances, items = self._knn.kneighbors(self._data.getcol(item_index).transpose(), return_distance=True)
            distances = distances[0][items[0] != item_index]
            items = items[0][items[0] != item_index]

            # update dictionary of distances for each item
            for item, distance in zip(items, distances):
                if item not in nearest_items:
                    nearest_items[item] = distance
                else:
                    nearest_items[item] = min(nearest_items[item], distance)

        # get items for recommendation
        sorted_items = sorted(nearest_items.items(), key=lambda x: x[1])[:items_count]
        return np.array(list(zip(*sorted_items))[0])

    def __str__(self) -> str:
        return f'Item based [k_nearest_neighbours = {self._k_nearest_neighbours}]'
