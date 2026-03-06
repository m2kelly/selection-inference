#runs specific for this project interactions with the database
#mysql handler is a general class for interacting with any mysql database
import pandas as pd
from mysql_handler import MysqlHandler

class DatabaseInterface (MysqlHandler):

    def __init__(self, host=None, user=None, password=None, database=None):
        super().__init__(host, user, password, database)
        self.create_index_table()

    def create_index_table(self):
        #create table if it does not exist
        #index table stores the name of tables given the selection coefficient and the scaler value, and if a mutation table (true) or
        #a substitution table (false)

        #check if the table exists
        self.cursor.execute('SHOW TABLES')
        tables = self.cursor.fetchall()

        if ('index_table',) in tables: return

        #create table
        self.cursor.execute('CREATE TABLE index_table (selection_coefficient VARCHAR(255), scaler VARCHAR(255), mutaion BOOLEAN, table_name VARCHAR(255), PRIMARY KEY (selection_coefficient, scaler, mutaion))')
        self.connection.commit()

    def insert_table_into_database(self, table, selection_coefficient, scaler, mutation,table_name=None):
        selection_coefficient = str(selection_coefficient); scaler = str(scaler)

        #if table_name is None, create a table name by hashing the parameters into a smaller string
        if table_name is None: table_name = str(hash((selection_coefficient, scaler, mutation)))
        #check if the type of the table is an instance of pandas dataframe
        if not isinstance(table, pd.DataFrame): table = pd.DataFrame(table)
        table.to_sql(name=table_name, con=self.engine, if_exists='replace', index=False)

        #insert into index table
        try:
            self.cursor.execute('INSERT INTO index_table VALUES (%s, %s, %s, %s)', (selection_coefficient, scaler, mutation, table_name))
            self.connection.commit()
        except Exception as e:
            print(e);
            return

    def get_tablename_from_index(self, selection_coefficient, scaler, mutation):
        selection_coefficient = str(selection_coefficient); scaler = str(scaler)
        self.cursor.execute('SELECT table_name FROM index_table WHERE selection_coefficient = %s AND scaler = %s AND mutaion = %s', (selection_coefficient, scaler, mutation))
        table_name = self.cursor.fetchall()
        return table_name[0][0]

    def get_table_from_index(self, selection_coefficient, scaler, mutation):
        table_name = self.get_tablename_from_index(selection_coefficient, scaler, mutation)
        if table_name is None: raise ValueError('Table does not exist')
        return self.get_table_as_pddf(table_name)
