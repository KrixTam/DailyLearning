from openai import OpenAI
import requests
from litellm import completion


def hello_world_01():
    your_api_key = input("请输入你的API Key：")
    client = OpenAI(api_key=your_api_key,
                    base_url="https://api.siliconflow.cn/v1")
    response = client.chat.completions.create(
        # model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        model="Qwen/Qwen3-8B",
        messages=[
            {'role': 'user',
            'content': "推理模型会给市场带来哪些新的机会"}
        ],
        stream=True
    )

    for chunk in response:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.choices[0].delta.reasoning_content:
            print(chunk.choices[0].delta.reasoning_content, end="", flush=True)


def hello_world_02():
    your_api_key = input("请输入你的API Key：")
    client = OpenAI(api_key=your_api_key,
                    base_url="https://api.siliconflow.cn/v1")
    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        # model="Qwen/Qwen3-8B",
        messages=[
            {'role': 'user',
            'content': "推理模型会给市场带来哪些新的机会"}
        ],
        stream=False
    )

    print(response.choices[0].message.content)


def hello_world_03():
    your_api_key = input("请输入你的API Key：")
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": "推理模型会给市场带来哪些新的机会"
            }
        ],
        "stream": False,
        "max_tokens": 4096,
        "thinking_budget": 4096,
        "min_p": 0.05,
        "stop": None,
        "temperature": 0.8,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    headers = {
        "Authorization": "Bearer " + your_api_key,
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)


def hello_world_04():
    your_api_key = input("请输入你的API Key：")
    response = completion(
        model="openai/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        custom_llm_provider="",
        messages=[
            {
                "role": "user",
                "content": "推理模型会给市场带来哪些新的机会"
            }
        ],
        stream=False,
        api_base="https://api.siliconflow.cn",
        api_key=your_api_key,
        max_tokens=4096,
        temperature=0.8,
        stop=None,
        frequency_penalty=0.5,
        top_p=0.7,
        min_p=0.05,
        top_k=50,
    )
    print(response.choices[0].message.content)


if __name__ == '__main__':
    # hello_world_01()
    # hello_world_02()
    # hello_world_03()
    hello_world_04()
