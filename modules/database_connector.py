import mysql.connector
from modules.log_create import mysql_log

class MySQLConnector:
    def __init__(self, host, user, password, database):
        self.__host = host
        self.__user = user
        self.__password = password
        self.__database = database
    
    def connect(self):
        try:
            return mysql.connector.connect(
                host = self.__host,
                user = self.__user,
                password = self.__password,
                database = self.__database
            )
        except Exception as e:
            mysql_log.logger.error(f'Error connecting to MySQL: {e}')
    
    def insert_data(self, table, data):
        connection = self.connect()
        cursor = connection.cursor()

        COLUMNS = ', '.join(data.keys())
        VALUES = ', '.join(['%s' for _ in data.values()])

        query = f"INSERT INTO {table} ({COLUMNS}) VALUES ({VALUES})"

        try:
            cursor.execute(query, tuple(data.values()))
            connection.commit()
            print('Data inserted successfully')
        except Exception as e:
            mysql_log.logger.warning(f'Error inserting data: {e}')
        finally:
            cursor.close()
        
    def get_latest_row(self, table):
        connection = self.connect()
        cursor = connection.cursor(dictionary=True)
        query = f"SELECT * FROM {table} ORDER BY si_id DESC LIMIT 1"
        try:
            cursor.execute(query)
            result = cursor.fetchone()
            return result
        except Exception as e:
            mysql_log.logger.warning(f'Error fetching latest row: {e}')
            return None
        finally:
            cursor.close()