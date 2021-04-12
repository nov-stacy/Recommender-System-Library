import typing as tp

import requests

URL = 'http://127.0.0.1:5000/change/'

memory_based_params = {'k_nearest_neighbours': 6}
sgd_params = {'dimension': 6, 'learning_rate': 0.0001, 'user_regularization': 0.2, 'item_regularization': 0.3}
dimension_params = {'dimension': 6}


models_params_dict: tp.Dict[str, tp.Dict[str, tp.Any]] = {
    1: memory_based_params,
    2: memory_based_params,
    3: dimension_params,
    4: dimension_params,
    5: sgd_params,
    6: dimension_params,
    7: sgd_params,
}


def main():

    for system_id in models_params_dict:
        params = {
            'params': models_params_dict[system_id]
        }
        result = requests.post(URL + str(system_id), json=params)
        result_print = 'OK' if result.status_code < 300 else f'ERROR'
        print(f'{system_id}: {result_print}')


if __name__ == '__main__':
    main()
