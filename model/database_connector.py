import mysql.connector
from model.log_create import mysql_log

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
    def select_data(self, table, condition=None):
        try:
            connection = self.connect()
            cursor = connection.cursor()
            query = f"SELECT * FROM {table}"
            if condition:
                query += f" WHERE {condition}"
            cursor.execute(query)
            # 取得欄位名稱
            columns = [column[0] for column in cursor.description]
            result = cursor.fetchall()
            
            # 將每筆資料轉換為字典形式，並與欄位名稱一一對應
            dict_list = [dict(zip(columns, row)) for row in result]
            return dict_list
        except Exception as e:
            mysql_log.logger.warning(f'Error selecting data: {e}')
            return []
    def update_data(self, table, data, condition):
        try:
            connection = self.connect()
            cursor = connection.cursor()
            set_clause = ", ".join([f"{key} = '{value}'" for key, value in data.items()])
            condition_clause = " AND ".join([f"{key} = '{value}'" for key, value in condition.items()])

            query = f"UPDATE {table} SET {set_clause} WHERE {condition_clause}"
            cursor.execute(query)
            connection.commit()
            print(f"Successfully updated data in {table}")
        except Exception as e:
            mysql_log.logger.warning(f'Error updating data: {e}')
    def update_last_record(self, table, update_data, timestamp_column):
        try:
            connection = self.connect()
            cursor = connection.cursor()
            query = f"SELECT {timestamp_column} FROM {table} ORDER BY {timestamp_column} DESC LIMIT 1"
            cursor.execute(query)
            result = cursor.fetchone()

            if result:
                last_record_timestamp = result[0]
                condition = {timestamp_column: last_record_timestamp}
                self.update_data(table, update_data, condition)
            else:
                print(f"No records found in {table}")

        except Exception as e:
            mysql_log.logger.warning(f'Error updating last record: {e}')
    def delete_data(self, table, condition):
        pass