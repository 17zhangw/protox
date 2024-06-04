import pglast
# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: MIT
#
"""This module provides a connection to the PostgreSQL database for benchmarking"""
import psycopg2
from connectors.connector import DBConnector
import configparser
import time
import os
import json
from utils.custom_logging import logger

FORCES = ["no_forceseq", "no_forceidx", "no_forceidxbase"]


def get_aliases(sql_query):
    tree = pglast.parse_sql(sql_query)
    aliases = [x.alias.aliasname.value if x.alias else x.relname.value for x in pglast.Node(tree)[0].traverse() if getattr(x, 'node_tag', None) == 'RangeVar']
    return aliases


class PostgresConnector(DBConnector):
    """This class handles the connection to the tested PostgreSQL database"""

    def __init__(self, config):
        super().__init__()
        # get connection config from config-file
        self.config = configparser.ConfigParser()
        self.config.read(config)
        defaults = self.config['DEFAULT']
        user = defaults['DB_USER']
        database = defaults['DB_NAME']
        password = defaults['DB_PASSWORD']
        host = defaults['DB_HOST']
        port = defaults['DB_PORT']
        self.timeout = defaults['TIMEOUT_MS']
        self.postgres_connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        self.connect()

        self.forceseq = False
        self.forceidx = False
        self.forceidxbase = False

    def connect(self) -> None:
        self.connection = psycopg2.connect(self.postgres_connection_string)
        self.cursor = self.connection.cursor()
        self.cursor.execute(f'set statement_timeout to {self.timeout}; commit;')

    def close(self) -> None:
        self.cursor.close()
        self.connection.close()

    def set_disabled_knobs(self, knobs: list) -> None:
        self.no_forceseq = False
        self.no_forceidx = False

        # todo enable all others rules before
        all_knobs = set(PostgresConnector.get_knobs())
        statements = ''
        for knob in all_knobs:
            if knob in FORCES:
                continue

            if knob not in knobs:
                statements += f'SET {knob} to ON;'
        for knob in knobs:
            if knob in FORCES:
                continue

            statements += f'SET {knob} to OFF;'

        while True:
            try:
                self.cursor.execute(statements)
                break
            except Exception as e:
                logger.warn(f"Error with setting knobs: {e}")

        # Disabling no force seq means force seq.
        self.forceseq = ("no_forceseq" in knobs)
        self.forceidx = ("no_forceidx" in knobs)
        self.forceidxbase = ("no_forceidxbase" in knobs)

    def distort_query(self, query):
        # Force sequential dominates force index.
        aliases = get_aliases(query)
        if self.forceseq:
            prefix = "/*+ " + " ".join([f"SeqScan({t})" for t in aliases]) + " Set(enable_nestloop OFF) */ "
            query = prefix + query
        elif self.forceidx:
            prefix = "/*+ " + " ".join([f"NoSeqScan({t})" for t in aliases]) + " Set(enable_hashjoin OFF) Set(enable_mergejoin OFF) */ "
            query = prefix + query
        elif self.forceidxbase:
            prefix = "/*+ " + " ".join([f"NoSeqScan({t})" for t in aliases]) + " */ "
            query = prefix + query

        logger.info(f"Preparing query {query}")
        return query

    def explain(self, query: str) -> str:
        """Explain a query and return the json query plan"""
        query = self.distort_query(query)
        self.cursor.execute(f'EXPLAIN (FORMAT JSON) {query}')
        return json.dumps(self.cursor.fetchone()[0][0]['Plan'])

    def execute(self, query: str) -> DBConnector.TimedResult:
        """Execute the query and return its result"""
        query = self.distort_query(query)
        begin = time.time_ns()
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        elapsed_time_usec = int((time.time_ns() - begin) / 1_000)

        return DBConnector.TimedResult(result, elapsed_time_usec)

    @staticmethod
    def get_name() -> str:
        return 'postgres'

    @staticmethod
    def get_knobs() -> list:
        """Static method returning all knobs defined for this connector"""
        with open(os.path.dirname(__file__) + '/../knobs/postgres.txt', 'r', encoding='utf-8') as f:
            return [line.replace('\n', '') for line in f.readlines()] + FORCES
