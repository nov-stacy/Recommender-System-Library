from create_recommender_system import models_params_dict

import requests


URL = 'http://127.0.0.1:5000/predict_items/'


models_indices = list(range(1, len(models_params_dict) + 1))
users_list = list(range(10))


def main():

    for system_id in models_indices:
        for user_index in users_list:
            params = {
                'user': user_index,
                'items_count': 5
            }
            result = requests.get(URL + str(system_id), json=params)
            result_print = result.json()['result'] if result.status_code < 300 else 'ERROR'
            print(f'{system_id}, {user_index}: {result_print}')


if __name__ == '__main__':
    main()
