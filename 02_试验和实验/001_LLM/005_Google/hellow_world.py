from google import genai


def hello_world(model_name):
    print(f"\n\n======  {model_name}  ======")

    response = client.models.generate_content(
        model=model_name, contents="请简单给我介绍一下你对大模型下关于Agent的理解。"
    )

    print(response.text)


if __name__ == '__main__':
    your_api_key = input("请输入你的API Key：")
    client = genai.Client(api_key=your_api_key)
    hello_world("gemini-2.5-flash")
    hello_world("gemini-2.5-flash-lite")
