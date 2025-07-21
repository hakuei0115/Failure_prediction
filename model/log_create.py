import logging

class Logger:
    def __init__(self, log_file, level, log_name):
        self.level = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        self.__log_level = self.level.get(level)
        # 設定 Logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(self.__log_level)

        # 設定輸出到檔案
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter(fmt='%(asctime).19s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 設定輸出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

error_log = Logger('log/error.log', 'warning', 'error_logger')
mysql_log = Logger('log/Mysql_error.log', 'warning', 'mysql_logger')
mqtt_log = Logger('log/Mqtt_error.log', 'warning', 'mqtt_logger')