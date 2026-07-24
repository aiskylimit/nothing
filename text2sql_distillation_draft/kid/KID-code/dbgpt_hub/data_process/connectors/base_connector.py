# -*- encoding: utf-8 -*-
from abc import ABC, abstractmethod


class BaseConnector(ABC):

    def __init__(
        self,
        host="127.0.0.1",
        port=3306,
        user=None,
        passwd=None,
        db=None,
        charset="utf8",
        *args,
        **kwargs
    ):
        self._host = host
        self._port = port
        self._user = user
        self._passwd = passwd
        self._db = db
        self._conn = None
        self._cursor = None

    def __del__(self):
        if self._cursor:
            self._cursor.close()

        if self._conn:
            self._conn.close()

    @abstractmethod
    def get_connect(self):
        pass

    @abstractmethod
    def get_cursor(self, cursor=None):
        pass

    @abstractmethod
    def select_db(self, db):
        pass

    @abstractmethod
    def get_all_tables(self, args=None):
        pass

    @abstractmethod
    def execute(self, sql, args=None):
        pass

    @abstractmethod
    def get_version(self, args=None):
        pass

    @abstractmethod
    def get_all_table_metadata(self, args=None):
        pass

    @abstractmethod
    def get_table_metadata(self, db, table, args=None):
        pass

    @abstractmethod
    def get_table_field_metadata(self, db, table, args=None):
        pass
