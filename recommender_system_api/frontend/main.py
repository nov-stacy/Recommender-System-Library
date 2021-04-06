import flask
from rest_framework import status

from recommender_system_api import backend

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def main() -> flask.Response:
    return flask.render_template('main.html')


@app.route('/create', methods=['POST'])
def create_recommender_system() -> flask.Response:
    try:
        id_ = backend.create_recommender_system(flask.request.json)
        return flask.make_response(flask.jsonify({'id': id_}), status.HTTP_201_CREATED)
    except (KeyError, TypeError, ValueError):
        return flask.Response(status=status.HTTP_400_BAD_REQUEST)


@app.route('/change/<system_id>', methods=['POST'])
def change_recommender_system(system_id: int) -> flask.Response:
    try:
        backend.change_recommender_system(system_id, flask.request.json)
        return flask.Response(status=status.HTTP_200_OK)
    except (KeyError, TypeError, ValueError):
        return flask.Response(status=status.HTTP_400_BAD_REQUEST)


@app.route('/train/<system_id>', methods=['POST'])
def train_recommender_system(system_id: int) -> flask.Response:
    pass


@app.route('/clear/<system_id>', methods=['POST'])
def clear_recommender_system(system_id: int) -> flask.Response:
    try:
        backend.clear_recommender_system(system_id)
        return flask.Response(status=status.HTTP_200_OK)
    except (KeyError, TypeError, ValueError):
        return flask.Response(status=status.HTTP_400_BAD_REQUEST)


@app.route('/delete/<system_id>', methods=['DELETE'])
def delete_recommender_system(system_id: int) -> flask.Response:
    try:
        backend.delete_recommender_system(system_id)
        return flask.Response(status=status.HTTP_200_OK)
    except (KeyError, TypeError, ValueError):
        return flask.Response(status=status.HTTP_400_BAD_REQUEST)


@app.route('/predict_ratings/<system_id>/<user_id>', methods=['GET'])
def get_predicted_ratings(system_id: int, user_id: int) -> flask.Response:
    pass


@app.route('/predict_items/<system_id>/<user_id>', methods=['GET'])
def get_predicted_items(system_id: int, user_id: int) -> flask.Response:
    pass


@app.route('/metric/<system_id>', methods=['GET'])
def get_metric_for_system(system_id: int) -> flask.Response:
    pass


if __name__ == '__main__':
    app.run(debug=True)
