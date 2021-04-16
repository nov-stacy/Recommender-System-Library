import typing as tp

import secrets
import sqlite3

from recommender_system_api.backend.work_with_database._settings import PATH_TO_DATABASE
from recommender_system_api.backend.work_with_database._file_system import check_path_exist


PATH_TO_DATABASE_TABLE_WITH_MODELS = f'{PATH_TO_DATABASE}/database.db'


def __create_connection() -> sqlite3.Connection:
    return sqlite3.connect(PATH_TO_DATABASE_TABLE_WITH_MODELS)


def __create_cursor(connection: sqlite3.Connection) -> sqlite3.Cursor:
    return connection.cursor()


def __create_database() -> None:

    sql_1 = 'CREATE TABLE systems (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER)'
    sql_2 = 'CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT NOT NULL)'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql_1)
        connection.commit()
        cursor.execute(sql_2)
        connection.commit()


def __check_table() -> None:

    if not check_path_exist(PATH_TO_DATABASE_TABLE_WITH_MODELS):
        __create_database()


def __check_token(token: str) -> bool:

    sql = f'SELECT id FROM users WHERE token = "{token}"'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        rows_count = len(cursor.fetchall())

    return rows_count == 0


def insert_new_model(user_id: int) -> int:

    __check_table()

    sql = f'INSERT INTO systems (user_id) VALUES ({user_id})'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        connection.commit()
        system_id = cursor.lastrowid

    return system_id


def insert_new_user() -> str:

    __check_table()

    token = secrets.token_urlsafe(16)
    while not __check_token(token):
        token = secrets.token_urlsafe(16)

    sql = f'INSERT INTO users (token) VALUES ("{token}")'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        connection.commit()

    return token


def check_model(system_id: int, user_id: int) -> bool:

    __check_table()

    sql = f'SELECT id FROM systems WHERE id = {system_id} and user_id = {user_id}'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        rows_count = len(cursor.fetchall())

    return rows_count == 1


def check_user(token: str) -> tp.Optional[int]:

    __check_table()

    sql = f'SELECT id FROM users WHERE token = "{token}"'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        result = cursor.fetchall()

    if len(result) != 1:
        return None

    return result[0][0]


def delete_model(system_id: int, user_id: int) -> None:

    __check_table()

    sql = f'DELETE FROM systems WHERE id = {system_id} AND user_id = {user_id}'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        connection.commit()
