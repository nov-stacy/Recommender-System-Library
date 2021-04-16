import typing as tp

import requests

URL = 'http://127.0.0.1:5000/create'

memory_based_params = {'k_nearest_neighbours': 8}
sgd_params = {'dimension': 8, 'learning_rate': 0.0001, 'user_regularization': 0.2, 'item_regularization': 0.3}
dimension_params = {'dimension': 8}


models_params_dict: tp.Dict[str, tp.Dict[str, tp.Any]] = {
    'user_based_model': memory_based_params,
    'item_based_model': memory_based_params,
    'latent_factor_als_model': dimension_params,
    'latent_factor_hals_model': dimension_params,
    'latent_factor_sgd_model': sgd_params,
    'latent_factor_svd_model': dimension_params,
    'implicit_sgd_model': sgd_params,
}


def main():

    for model_type in models_params_dict:
        headers = {
            'token': 'epvL3zUwMhZagI7HA5XBvA'
        }
        params = {
            'type': model_type,
            'params': models_params_dict[model_type]
        }
        result = requests.post(URL, json=params, headers=headers)
        result_print = result.json()["id"] if result.status_code < 300 else f'ERROR: {result.status_code}'
        print(f'{model_type}: {result_print}')


if __name__ == '__main__':
    main()
