from litellm import completion, CustomLLM


def hello_world():
    response = completion(
        model="ollama/qwen3:8b",
        custom_llm_provider="",
        messages=[
            {
                "role": "user",
                "content": "推理模型会给市场带来哪些新的机会"
            }
        ],
        stream=False,
        api_base="http://localhost:11434",
        max_tokens=4096,
        temperature=0.8,
        stop=None,
        frequency_penalty=0.5,
        top_p=0.7
    )
    print(response.choices[0].message.content)


if __name__ == '__main__':
    hello_world()
