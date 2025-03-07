# 函数调用的示例（不进行转义的情况）
import ollama
from utils import timer
import requests
from moment import moment
from urllib.parse import urlencode
key = 'ZQ4BZ-7CT6T-6LKXD-VCVLZ-OH5QH-FTFCQ'


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

@timer
def demo():
    available_functions = {
        'get_weather': get_weather,
    }
    messages_01 = [
        {
            'role': 'user',
            # 'content': '请问佛山市南海区今天的天气怎么样？'
            'content': '请问广州市今天天气如何？会下雨吗？'
        }
    ]
    res_01 = ollama.chat(
        model='qwen2.5:7b',
        messages=messages_01,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': '通过地名查找当地目前的天气',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'address': {
                                'type': 'string',
                                'description': '地名',
                            },
                        },
                        'required': ['address'],
                    },
                },
            },
        ],
    )
    # print(response['message']['tool_calls'])
    for tool in res_01['message']['tool_calls'] or []:
        function_to_call = available_functions[(tool['function']['name'])]
        if function_to_call:
            function_name = tool['function']['name']
            # print('Calling function:', function_name)
            # print('Arguments:', tool['function']['arguments'])
            function_response = function_to_call(**tool['function']['arguments'])
            print('Function output:', function_response)
            messages_01.append(res_01['message'])
            messages_01.append({
                'role': 'function',
                'name': function_name,
                'content': function_response
            })
            res_02 = ollama.chat(
                # model='qwen2.5:7b',
                # model='mistral',
                model='deepseek-r1:7b',
                messages=messages_01,
            )
            print(res_02['message']['content'])
        else:
            print('Function', tool['function']['name'], 'not found')


if __name__ == '__main__':
    demo()
