from create_recommender_system import models_params_dict

import requests


URL = 'http://127.0.0.1:5000/clear/'


models_indices = list(range(1, len(models_params_dict) + 1))


def main():

    for system_id in models_indices:
        result_print = 'OK' if requests.post(URL + str(system_id)).status_code < 300 else 'ERROR'
        print(f'{system_id}: {result_print}')


if __name__ == '__main__':
    main()
