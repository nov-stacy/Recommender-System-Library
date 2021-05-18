import typing as tp

import secrets
import sqlite3

from backend.work_with_database._settings import PATH_TO_DATABASE
from backend.work_with_database._work_with_file_system import check_path_exist


__all__ = [
    'insert_new_model_into_table', 'insert_new_user_into_table',
    'check_model_in_table', 'check_user_in_table', 'delete_model_from_table'
]


PATH_TO_DATABASE_TABLE_WITH_MODELS = f'{PATH_TO_DATABASE}/database.db'


def _create_connection() -> sqlite3.Connection:
    return sqlite3.connect(PATH_TO_DATABASE_TABLE_WITH_MODELS)


def _create_cursor(connection: sqlite3.Connection) -> sqlite3.Cursor:
    return connection.cursor()


def _create_database() -> None:

    sql_1 = 'CREATE TABLE systems (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER)'
    sql_2 = 'CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT NOT NULL)'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql_1)
        connection.commit()
        cursor.execute(sql_2)
        connection.commit()


def _check_table() -> None:
    if not check_path_exist(PATH_TO_DATABASE_TABLE_WITH_MODELS):
        _create_database()


def _check_token(token: str) -> bool:

    sql = f'SELECT id FROM users WHERE token = "{token}"'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql)
        rows_count = len(cursor.fetchall())

    return rows_count == 0


def insert_new_model_into_table(user_id: int) -> int:

    _check_table()

    sql = f'INSERT INTO systems (user_id) VALUES ({user_id})'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql)
        connection.commit()
        system_id = cursor.lastrowid

    return system_id


def insert_new_user_into_table() -> str:

    _check_table()

    token = secrets.token_urlsafe(16)
    while not _check_token(token):
        token = secrets.token_urlsafe(16)

    sql = f'INSERT INTO users (token) VALUES ("{token}")'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql)
        connection.commit()

    return token


def check_model_in_table(user_id: int, system_id: int) -> bool:

    _check_table()

    sql = f'SELECT id FROM systems WHERE id = {system_id} and user_id = {user_id}'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql)
        rows_count = len(cursor.fetchall())

    return rows_count == 1


def check_user_in_table(token: str) -> tp.Optional[int]:

    _check_table()

    sql = f'SELECT id FROM users WHERE token = "{token}"'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql)
        result = cursor.fetchall()

    if len(result) != 1:
        return None
    return result[0][0]


def delete_model_from_table(user_id: int, system_id: int) -> None:

    _check_table()

    sql = f'DELETE FROM systems WHERE id = {system_id} AND user_id = {user_id}'

    with _create_connection() as connection:
        cursor = _create_cursor(connection)
        cursor.execute(sql)
        connection.commit()
