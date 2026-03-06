import mysql.connector
import pandas as pd
from sqlalchemy import create_engine


class MysqlHandler ():

    def __init__(self, host=None, user=None, password=None, database=None):
        if host is None:
            #if any other parameter is not None, raise error
            if any([user, password, database]):
                raise ValueError('If host is None, all other parameters must be None')
            host = 'localhost'
            user = 'CBaSE'
            password = 'darwin1809'
            database = 'CBaSE'
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        self.engine = None

        self.connect()

    def connect(self):
        self.connection = mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        self.cursor = self.connection.cursor()
        self.engine = create_engine('mysql+mysqlconnector://%s:%s@%s/%s' % (self.user,
                                                                            self.password,
                                                                            self.host,
                                                                            self.database),
                                    echo=False)

    def get_table_as_pddf(self, table_name):
        self.cursor.execute('SELECT * FROM `%s`' % table_name)
        table = self.cursor.fetchall()
        return pd.DataFrame(table)

    def get_all_table_names(self):
        self.cursor.execute('SHOW TABLES')
        tables = self.cursor.fetchall()
        return list(tables)

    def insert_table_into_database(self, table, table_name):
        #check if the type of the table is an instance of pandas dataframe
        if not isinstance(table, pd.DataFrame): table = pd.DataFrame(table)
        table.to_sql(name=table_name, con=self.engine, if_exists='replace', index=False)

    def delete_all_tables(self):
        tables = self.get_all_table_names()
        for table in tables:
            self.cursor.execute('DROP TABLE `%s`' % table)
        self.connection.commit()
