import codecs
import pickle

import numpy as np

from create_recommender_system import models_params_dict

import requests

from recommender_system_library.extra_functions.work_with_train_data import read_matrix_from_file

URL = 'http://127.0.0.1:5000/metric'
PATH_TO_DATA = '../data_train/matrices/random_matrix_10.npz'

models_indices = list(range(1, len(models_params_dict) + 1))
metrics = ['precision', 'recall']
users_list = list(range(10))
predict_params = {
    'k_items': list(range(10)),
    'barrier_value': list(np.arange(0.1, 1, 0.1))
}


def main():

    test_matrix = read_matrix_from_file(PATH_TO_DATA)

    for system_id in models_indices:
        for metric in metrics:
            for parameter in predict_params:
                for parameter_value in predict_params[parameter]:

                    params = {
                        'test_data': codecs.encode(pickle.dumps(test_matrix), 'base64').decode(),
                        'users': users_list,
                        'predict_parameter_name': parameter,
                        'predict_parameter_value': parameter_value,
                    }

                    result = requests.get(f'{URL}/{metric}/{system_id}', json=params)
                    result_print = result.json()['result'] if result.status_code < 300 else 'ERROR'
                    print(f'{system_id}, metric: {metric}, parameter: {parameter}, '
                          f'parameter_value: {parameter_value}, result: {result_print}')


if __name__ == '__main__':
    main()
