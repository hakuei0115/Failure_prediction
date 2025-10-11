import os
import json
import time
import mysql.connector
from scipy.stats import norm
from dotenv import load_dotenv
from config.constants import *

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

LIFESPAN_DB = os.getenv("POLICY_DB")
LIFESPAN_TABLE = os.getenv("LIFESPAN_TABLE")

DATA_DB = os.getenv("DATA_DB")
LIFE_LIMIT_TABLE = os.getenv("LIFE_LIMIT_TABLE")

def lifespan_estimation(do_count, Y=500000, mean_life=10000000, std_life=5000000):
    p_x = norm.cdf(do_count, loc=mean_life, scale=std_life)
    p_xY = norm.cdf(do_count + Y, loc=mean_life, scale=std_life)
    P_T_gte_x = 1 - p_x
    P_T_gte_xY = 1 - p_xY
    cond_prob = P_T_gte_xY / P_T_gte_x if P_T_gte_x > 0 else 0
    return f"已運作 {do_count:,} 次後，還能再 {Y:,} 次的機率：約 {cond_prob:.2%}"

def decode_valve_state(do_brk):
    bits = bin(do_brk)[2:].zfill(48)
    channels = [
        20, 19, 18, 17,
        24, 23, 22, 21,
        12, 11, 10, 9,
        16, 15, 14, 13,
        4, 3, 2, 1,
        8, 7, 6, 5
    ]

    state_dict = {}
    for i, ch in enumerate(channels):
        pair = bits[i * 2:(i + 1) * 2]
        if pair == "11":
            state = "ON"
        elif pair == "00":
            state = "OFF"
        else:
            state = "異常"
        state_dict[f"ch_{ch}"] = state

    return json.dumps(state_dict, ensure_ascii=False)

def main():
    sql = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DATA_DB
    )
    sql_life = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=LIFESPAN_DB
    )

    cursor = sql.cursor(dictionary=True)
    cursor_life = sql_life.cursor()

    valve_islands = [
        "192.168.17.30", "192.168.17.31", "192.168.17.32",
        "192.168.17.33", "192.168.17.34", "192.168.17.36"
    ]

    for ip in valve_islands:
        query = f"SELECT * FROM {LIFE_LIMIT_TABLE} WHERE si_ip = %s ORDER BY si_ts DESC LIMIT 1"
        cursor.execute(query, (ip,))
        result = cursor.fetchone()

        if not result:
            print(f"閥島 {ip} 沒有資料")
            continue

        do_counts = {k: v for k, v in result.items() if k.startswith("do_count_")}
        do_brk = result["do_brk"]

        estimation_dict = {}
        for k, v in do_counts.items():
            idx = k.split("_")[2]
            estimation_dict[f"valve_{idx}"] = lifespan_estimation(v)

        estimation_json = json.dumps(estimation_dict, ensure_ascii=False)
        do_brk_json = decode_valve_state(do_brk)

        insert_sql = f"""
            INSERT INTO {LIFESPAN_TABLE} (ip, estimation, do_brk_state, sync)
            VALUES (%s, %s, %s, %s)
        """
        cursor_life.execute(insert_sql, (ip, estimation_json, do_brk_json, 0))
        sql_life.commit()

        print(f"閥島 {ip} 已完成壽限估算與狀態解析並寫入資料庫")

    cursor.close()
    cursor_life.close()
    sql.close()
    sql_life.close()


if __name__ == "__main__":
    while True:
        main()
        time.sleep(10)
