import typing as tp

import flask

from recommender_system_api import backend

app = flask.Flask(__name__)


def method_for_exception_catch(good_status_code):
    """
    Method for creating a decorator
    """

    def decorator(function):
        """
        Decorator for determining the behavior of the program when an error occurs
        """

        def wrapper(*args, **kwargs):
            try:
                result = function(*args, **kwargs)  # request processing
                if result is None:  # if the function returns no response, the method returns only the code
                    return flask.Response(status=good_status_code)
                return flask.make_response(flask.jsonify(result), good_status_code)
            # error accessing class methods or invalid user data
            except (AttributeError, backend.AuthTokenError) as error:
                return flask.make_response(flask.jsonify({'message': str(error)}), 405)
            except Exception as error:
                return flask.make_response(flask.jsonify({'message': str(error)}), 400)

        wrapper.__name__ = function.__name__
        return wrapper

    return decorator


@app.route('/', methods=['GET'])
def main() -> flask.Response:
    return flask.render_template('main.html')


@app.route('/registration', methods=['POST'])
@method_for_exception_catch(201)
def registration_user() -> tp.Dict[str, tp.Any]:
    return backend.registration_user()


@app.route('/create', methods=['POST'])
@method_for_exception_catch(201)
def create_recommender_system() -> tp.Dict[str, tp.Any]:
    user_id = backend.check_user_token(flask.request.headers)
    return backend.create_recommender_system(user_id, flask.request.json)


@app.route('/change/<system_id>', methods=['POST'])
@method_for_exception_catch(204)
def change_recommender_system(system_id: int) -> None:
    user_id = backend.check_user_token(flask.request.headers)
    backend.change_recommender_system(user_id, int(system_id), flask.request.json)


@app.route('/train/<system_id>', methods=['POST'])
@method_for_exception_catch(202)
def train_recommender_system(system_id: int) -> None:
    user_id = backend.check_user_token(flask.request.headers)
    backend.train_recommender_system(user_id, int(system_id), flask.request.json)


@app.route('/status/<system_id>', methods=['POST'])
@method_for_exception_catch(200)
def check_status(system_id: int) -> tp.Dict[str, tp.Any]:
    user_id = backend.check_user_token(flask.request.headers)
    return backend.check_status_of_recommender_system(user_id, int(system_id))


@app.route('/clear/<system_id>', methods=['POST'])
@method_for_exception_catch(204)
def clear_recommender_system(system_id: int) -> None:
    user_id = backend.check_user_token(flask.request.headers)
    backend.clear_recommender_system(user_id, int(system_id))


@app.route('/delete/<system_id>', methods=['DELETE'])
@method_for_exception_catch(204)
def delete_recommender_system(system_id: int) -> None:
    user_id = backend.check_user_token(flask.request.headers)
    backend.delete_recommender_system(user_id, int(system_id))


@app.route('/predict_ratings/<system_id>', methods=['GET'])
@method_for_exception_catch(200)
def get_predicted_ratings(system_id: int) -> tp.Dict[str, tp.Any]:
    user_id = backend.check_user_token(flask.request.headers)
    return backend.get_list_of_ratings_from_recommender_system(user_id, int(system_id), flask.request.json)


@app.route('/predict_items/<system_id>', methods=['GET'])
@method_for_exception_catch(200)
def get_predicted_items(system_id: int) -> tp.Dict[str, tp.Any]:
    user_id = backend.check_user_token(flask.request.headers)
    return backend.get_list_of_items_from_recommender_system(user_id, int(system_id), flask.request.json)


@app.route('/metric/<metric_name>/<system_id>', methods=['GET'])
@method_for_exception_catch(200)
def get_metric_for_system(metric_name: str, system_id: int) -> tp.Dict[str, tp.Any]:
    user_id = backend.check_user_token(flask.request.headers)
    return backend.get_metric_for_recommender_system(user_id, int(system_id), metric_name, flask.request.json)


if __name__ == '__main__':
    app.run(debug=True)
