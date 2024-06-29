import ollama


res = ollama.chat(
    model='llava',
    messages=[
        {
            'role': 'user',
            'content': '请输出你识别出的文字文本',
            'images': ['/Users/krix/Downloads/test_02.jpg']
        }
    ]
)

print(res['message']['content'])
