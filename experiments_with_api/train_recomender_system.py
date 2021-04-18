import codecs

import pickle
import typing as tp

import requests

from recommender_system_library.extra_functions.work_with_train_data import read_matrix_from_file, get_train_matrix

URL = 'http://127.0.0.1:5000/train/'
PATH_TO_DATA = '../data_train/matrices/random_matrix_10.npz'

empty_train_params = {}
embedding_train_params = {'epochs': 10}

models_params_dict: tp.Dict[str, tp.Dict[str, tp.Any]] = {
    1: empty_train_params,
    2: empty_train_params,
    3: embedding_train_params,
    4: embedding_train_params,
    5: embedding_train_params,
    6: empty_train_params,
    7: embedding_train_params,
}


def main():

    train_matrix = read_matrix_from_file(PATH_TO_DATA)
    x_train_matrix = get_train_matrix(train_matrix, 0.3)

    headers = {
        'token': '<TOKEN>'
    }

    for system_id in models_params_dict:
        params = {
            'params': models_params_dict[system_id],
            'train_data': codecs.encode(pickle.dumps(x_train_matrix), 'base64').decode()
        }
        result = requests.post(URL + str(system_id), json=params, headers=headers)
        result_print = 'OK' if result.status_code < 300 else f'{result.status_code}: {result.json()}'
        print(f'{system_id}: {result_print}')


if __name__ == '__main__':
    main()
