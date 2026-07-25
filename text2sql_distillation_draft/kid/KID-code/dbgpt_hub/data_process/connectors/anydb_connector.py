from typing import Any
from .mysql_connector import MySQLConnector


class AnyDBConnector(BaseConnector):
    def __init__(
        self,
        db_type: str = "mysql",
        host: str = "127.0.0.1",
        port: str = 3306,
        user: Optional[str] = None,
        passwd: Optional[str] = None,
        db: Optional[str] = None,
        charset: str = "utf8",
        *args,
        **kwargs
    ) -> Any:
        super().__init__(db_type, host, port, user, passwd, db, charset, args, kwargs)
        if self.db_type == "mysql":
            self.connector = MySQLConnector(
                host=self.host, port=self.port, user=self.user, passwd=self.passwd
            )
        """TO DO: postgres, bigquery, etc."""

    def __del__(self) -> Any:
        super().__del__()

    def get_connect(self) -> Any:
        return self.connector.get_connect()

    def get_cursor(self, cursor=None):
        return self.connector.get_cursor()

    def select_db(self, db):
        return self.connector.select_db()

    def get_all_tables(self, args=None):
        return self.connector.get_all_tables(args)

    def execute(self, sql, args=None):
        return self.connector.execute(sql, args)

    def get_version(self, args=None):
        return self.connector.execute(args)

    def get_all_table_metadata(self, args=None):
        return self.connector.get_all_table_metadata(args)

    def get_table_metadata(self, db, table, args=None):
        return self.connector.get_table_metadata(db, table, args)

    def get_table_field_metadata(self, db, table, args=None):
        return self.connector.get_table_field_metadata(db, table, args)
