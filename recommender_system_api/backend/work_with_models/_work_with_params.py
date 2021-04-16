import codecs
import pickle
import typing as tp


__all__ = [
    'split_data',
    'merge_data',
    'get_data'
]

from scipy import sparse


def split_data(data: tp.Dict[str, tp.Any], parameters: tp.List[str]) -> tp.List[tp.Any]:

    if set(data.keys()) != set(parameters):
        raise ValueError
    return [data[parameter] for parameter in data]


def merge_data(params_names: tp.List[str], values: tp.List[tp.Any]) -> tp.Dict[str, tp.Any]:

    return dict(zip(params_names, values))


def get_data(data_str: str) -> sparse.coo_matrix:
    return pickle.loads(codecs.decode(data_str.encode(), 'base64'))
