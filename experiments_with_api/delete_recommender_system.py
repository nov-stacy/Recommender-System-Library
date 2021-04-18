from create_recommender_system import models_params_dict

import requests


URL = 'http://127.0.0.1:5000/delete/'


models_indices = list(range(1, len(models_params_dict) + 1))


def main():

    headers = {
        'token': '<TOKEN>'
    }

    for system_id in models_indices:
        result = requests.delete(URL + str(system_id), headers=headers)
        result_print = 'OK' if result.status_code < 300 else f'{result.status_code}: {result.json()}'
        print(f'{system_id}: {result_print}')


if __name__ == '__main__':
    main()
