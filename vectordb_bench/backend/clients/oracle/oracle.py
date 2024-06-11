import array
import logging
from contextlib import contextmanager
from typing import Optional, Tuple

import oracledb

from vectordb_bench.backend.clients.oracle.config import OracleDBCaseConfig

from ..api import VectorDB

log = logging.getLogger(__name__)


class Oracle(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: OracleDBCaseConfig,
        collection_name: str = "docs",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.collection_name = collection_name

        self.name = "Oracle"

        self._table_name = collection_name
        self._index_name = "doc_idx"
        self._id_col_name = "doc_id"
        self._vec_col_name = "doc_vec"

        self.conn, self.cursor = self._create_connection(**self.db_config)

        log.info(f"{self.name} config values: {self.db_config}\n{self.db_case_config}")

        if drop_old:
            self._drop_table()
            self._create_table(dim)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs):
        conn = oracledb.connect(**kwargs)
        cursor = conn.cursor()
        return conn, cursor

    def _drop_table(self):
        log.info(f"{self.name} client drop table : {self._table_name}")
        plsql_block = """
        DECLARE
            v_table_count INTEGER;
        BEGIN
            -- Check if the table exists
            SELECT COUNT(*) INTO v_table_count
            FROM user_tables
            WHERE table_name = :table_name;
            
            -- If the table exists, drop it
            IF v_table_count > 0 THEN
                EXECUTE IMMEDIATE 'DROP TABLE ' || :table_name;
            END IF;
        END;
        """
        self.cursor.execute(plsql_block, table_name=self._table_name.upper())

    def _create_table(self, dim: int):
        log.info(f"{self.name} client create table : {self._table_name}")
        create_table_sql = f"""
        CREATE TABLE {self._table_name} (
            {self._id_col_name} NUMERIC,
            {self._vec_col_name} VECTOR({self.dim})
        )
        """
        log.info(create_table_sql)
        self.cursor.execute(create_table_sql)

    @contextmanager
    def init(self) -> None:
        self.conn, self.cursor = self._create_connection(**self.db_config)
        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def insert_embeddings(
        self, embeddings: list[list[float]], metadata: list[int], **kwargs
    ) -> Tuple[int, Optional[Exception]]:
        try:
            insert_sql = f"""
            INSERT INTO {self._table_name} ({self._id_col_name}, {self._vec_col_name})
            VALUES (:1, :2)
            """
            parameters = [
                (metadata[i], array.array("f", embeddings[i]))
                for i in range(len(metadata))
            ]
            self.cursor.executemany(insert_sql, parameters)
            self.conn.commit()
            return len(metadata), None
        except Exception as e:
            log.warning(
                f"Failed to insert data into table ({self._table_name}), error: {e}"
            )
            return 0, e

    def search_embedding(
        self, query: list[float], k: int = 100, filters: dict | None = None
    ) -> list[int]:
        metric = self.db_case_config.search_param()['metric']
        select_sql = f"""
        SELECT {self._id_col_name}
        FROM {self._table_name}
        ORDER BY VECTOR_DISTANCE({self._vec_col_name}, :1, {metric})
        FETCH APPROXIMATE FIRST {k} ROWS ONLY
        """
        log.info(select_sql)
        self.cursor.execute(select_sql, (array.array("f", query),))
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]

    def optimize(self):
        self._create_index()

    def _create_index(self):
        log.info(f"{self.name} client create index : {self._index_name}")
        distance = self.db_case_config.index_param()["distance"]
        create_vector_index_sql = f"""
        CREATE VECTOR INDEX {self._index_name} ON {self._table_name} ({self._vec_col_name}) ORGANIZATION NEIGHBOR PARTITIONS
        DISTANCE {distance}
        WITH TARGET ACCURACY 95
        """
        log.info(create_vector_index_sql)
        self.cursor.execute(create_vector_index_sql)

    def ready_to_load(self):
        pass
