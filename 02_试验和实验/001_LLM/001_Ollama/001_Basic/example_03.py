# 函数调用的示例（不进行转义的情况）
import ollama
from utils import timer, get_weather


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
