import uuid
from typing import Any, Dict, List

import psycopg
from psycopg import sql

from .base import StorageAdapter


class PostgresAdapter(StorageAdapter):
    def __init__(self, host: str, dbname: str, user: str, pw: str, port: int):
        try:
            self.conn = psycopg.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=pw,
                port=port,
            )
        except psycopg.Error as e:
            raise Exception(f"Connection to DB not successful: {e}")

    def save_feedback(self, table: str, data: List[Dict[str, Any]]) -> bool:
        try:
            with self.conn.cursor() as cur:
                columns = data[0].keys()
                values = [[row[col] for col in columns] for row in data]

                query = sql.SQL(
                    """
                    INSERT INTO {table} ({columns})
                    VALUES ({placeholders})
                    ON CONFLICT (id) DO NOTHING
                """
                ).format(
                    table=sql.Identifier(table),
                    columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
                    placeholders=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
                )

                cur.executemany(query, values)
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception("Error inserting feedback data: {e}")

    def get_feedback(self, table: str, feedback_id: str) -> Dict[str, Any]:
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL(
                    """
                    SELECT
                        *
                    FROM {table}
                    WHERE id = %s
                """
                ).format(table=sql.Identifier(table))

                res = cur.execute(query, feedback_id)
                return res
        except Exception as e:
            raise Exception("Error querying feedback data: {e}")

    def get_feedback_all(self, table: str) -> List[Dict[str, Any]]:
        try:
            with self.conn.cursor() as cur:
                query = sql.SQL(
                    """
                    SELECT
                        *
                    FROM {table}
                """
                ).format(table=sql.Identifier(table))

                res = cur.execute(query)
                return res.fetchone()
        except Exception as e:
            raise Exception("Error querying all feedback data: {e}")
