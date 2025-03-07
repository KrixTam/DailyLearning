from time import perf_counter
import requests
from moment import moment
from urllib.parse import urlencode
key = 'ZQ4BZ-7CT6T-6LKXD-VCVLZ-OH5QH-FTFCQ'


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} 耗时：{end - start:.6f} 秒")
        return result
    return wrapper


def addr2loc(address: str) -> str:
    req_url = 'https://apis.map.qq.com/ws/geocoder/v1/?' + urlencode({'address': address, 'key': key, 'policy': 1})
    # print(req_url)
    response = requests.get(req_url)
    if response.status_code == 200:
        data = response.json()
        # print(data)
        if data['status'] == 0:
            return data['result']['ad_info']['adcode']
        else:
            print(data)
            return ""
    else:
        # print(response.json())
        return ""


def get_weather(address: str) -> str:
    '''
    通过地名查找当地目前的天气

    Args:
        address: 地名

    Returns:
        str: 当地天气情况
    '''
    adcode = addr2loc(address)
    if adcode == "":
        result = {'address': address, 'error': '不好意思，没有找到该地的天气情况。'}
    else:
        req_url = 'https://apis.map.qq.com/ws/weather/v1/?' + urlencode({'adcode': adcode, 'key': key})
        response = requests.get(req_url)
        if response.status_code == 200:
            data = response.json()
            # print(data)
            result = {
                '地址': address,
                '时间': moment().format('YYYY-MM-DD'),
                '天气': data['result']['realtime'][0]['infos']['weather'],
                '实时温度': data['result']['realtime'][0]['infos']['temperature'],
                '风向': data['result']['realtime'][0]['infos']['wind_direction'],
                '风力': data['result']['realtime'][0]['infos']['wind_power'],
                '湿度': data['result']['realtime'][0]['infos']['humidity']
            }
        else:
            # print(response.json())
            result = {'address': address, 'error': '不好意思，系统繁忙，未能找到该地的天气情况。'}
    return str(result)

# addr2loc('广州市')
# get_weather('广州市')
