import uuid
from contextlib import contextmanager
from typing import Generator

import psycopg


@contextmanager
def get_db_conn() -> Generator[psycopg.Connection, None, None]:
    """Context manager for database connections."""
    conn = None
    try:
        conn = psycopg.connect(
            host="localhost",
            dbname="postgres",
            user="postgres",
            password="postgres",
            port=5432,
        )
        yield conn
    except psycopg.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()


def save_to_db(conn: psycopg.Connection, pref_data: list[dict]):
    try:
        with conn.cursor() as cur:
            columns = pref_data[0].keys()
            placeholders = ",".join(["%s"] * len(columns))
            insert_query = f"""
                INSERT INTO preference ({','.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (id) DO NOTHING
            """

            values = [[row[col] for col in columns] for row in pref_data]
            cur.executemany(insert_query, values)
            conn.commit()
    except Exception:
        conn.rollback()
        raise
