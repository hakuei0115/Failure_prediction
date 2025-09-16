import requests
from modules.log_create import error_log

def send_sms(username, password, api_url, mobile: str, message: str):
    params = {
        "username": username,
        "password": password,
        "mobile": mobile,
        "message": message,
    }

    try:
        response = requests.get(api_url, params=params, timeout=10)
        data = response.json()
        
        if data.get("code") == "00000":
            print(f"簡訊發送成功，msgid={data['msgid']}")
        else:
            error_log.logger.error(f"簡訊發送失敗，錯誤碼={data.get('code')}，訊息={data.get('text')}")
    except Exception as e:
        error_log.logger.error(f"發送時發生錯誤: {e}")