import sqlite3

from recommender_system_api.backend.work_with_database._settings import PATH_TO_DATABASE
from recommender_system_api.backend.work_with_database._file_system import check_path_exist

PATH_TO_DATABASE_TABLE = f'{PATH_TO_DATABASE}/database.db'


def __create_connection() -> sqlite3.Connection:
    return sqlite3.connect(PATH_TO_DATABASE_TABLE)


def __create_cursor(connection: sqlite3.Connection) -> sqlite3.Cursor:
    return connection.cursor()


def __create_table() -> None:

    sql = 'CREATE TABLE systems (id INTEGER PRIMARY KEY AUTOINCREMENT)'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        connection.commit()


def __check_table() -> None:

    if not check_path_exist(PATH_TO_DATABASE_TABLE):
        __create_table()


def insert_new_system() -> int:

    __check_table()

    sql = 'INSERT INTO systems DEFAULT VALUES'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        connection.commit()
        system_id = cursor.lastrowid

    return system_id


def check_system(system_id: int) -> bool:

    __check_table()

    sql = f'SELECT id FROM systems WHERE id = {system_id}'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        rows_count = len(cursor.fetchall())

    return rows_count == 1


def delete_system(system_id: int) -> None:

    __check_table()

    sql = f'DELETE FROM systems WHERE id = {system_id}'

    with __create_connection() as connection:
        cursor = __create_cursor(connection)
        cursor.execute(sql)
        connection.commit()
