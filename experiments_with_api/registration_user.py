import requests

URL = 'http://127.0.0.1:5000/registration'


def main():
    result = requests.post(URL)
    result_print = result.json()['token'] if result.status_code < 300 else 'ERROR'
    print(f'result: "{result_print}"')


if __name__ == '__main__':
    main()
