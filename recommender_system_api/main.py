import typing as tp

import flask

from recommender_system_api import backend

app = flask.Flask(__name__)


def method_for_exception_catch(good_status_code):

    def decorator(function):

        def wrapper(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
                if result is None:
                    return flask.Response(status=good_status_code)
                return flask.make_response(flask.jsonify(result), good_status_code)
            except (TypeError, ValueError):
                return flask.Response(status=400)
            except NotImplementedError:
                return flask.Response(status=405)

        wrapper.__name__ = function.__name__
        return wrapper

    return decorator


@app.route('/', methods=['GET'])
def main() -> flask.Response:
    return flask.render_template('main.html')


@app.route('/create', methods=['POST'])
@method_for_exception_catch(201)
def create_recommender_system() -> tp.Dict[str, tp.Any]:
    return backend.create_recommender_system(flask.request.json)


@app.route('/change/<system_id>', methods=['POST'])
@method_for_exception_catch(204)
def change_recommender_system(system_id: int) -> None:
    backend.change_recommender_system(system_id, flask.request.json)


@app.route('/train/<system_id>', methods=['POST'])
@method_for_exception_catch(202)
def train_recommender_system(system_id: int) -> None:
    backend.train_recommender_system(system_id, flask.request.json)


@app.route('/status/<system_id>', methods=['POST'])
@method_for_exception_catch(200)
def check_status(system_id: int) -> tp.Dict[str, tp.Any]:
    return backend.check_status(system_id)


@app.route('/clear/<system_id>', methods=['POST'])
@method_for_exception_catch(204)
def clear_recommender_system(system_id: int) -> None:
    backend.clear_recommender_system(system_id)


@app.route('/delete/<system_id>', methods=['DELETE'])
@method_for_exception_catch(204)
def delete_recommender_system(system_id: int) -> None:
    backend.delete_recommender_system(system_id)


@method_for_exception_catch(200)
@app.route('/predict_ratings/<system_id>', methods=['GET'])
def get_predicted_ratings(system_id: int) -> tp.Dict[str, tp.Any]:
    return backend.get_list_of_ratings_from_recommender_system(system_id, flask.request.json)


@app.route('/predict_items/<system_id>', methods=['GET'])
@method_for_exception_catch(200)
def get_predicted_items(system_id: int) -> tp.Dict[str, tp.Any]:
    return backend.get_list_of_items_from_recommender_system(system_id, flask.request.json)


@app.route('/metric/<metric_name>/<system_id>', methods=['GET'])
@method_for_exception_catch(200)
def get_metric_for_system(metric_name: str, system_id: int) -> tp.Dict[str, tp.Any]:
    return backend.get_metric_for_recommender_system(metric_name, system_id, flask.request.json)


if __name__ == '__main__':
    app.run(debug=True)
