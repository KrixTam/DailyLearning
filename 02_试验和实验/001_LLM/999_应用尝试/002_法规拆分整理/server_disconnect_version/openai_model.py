import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional
# from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type,
# )
from tokenizer import Tokenizer
from limitter import RPM, TPM
from litellm import acompletion


@dataclass
class OpenAIModel:
    model_name: str = "Qwen/Qwen3-8B"
    api_key: str = None
    base_url: str = "https://api.siliconflow.cn"
    # model_name: str = "qwen3:8b"
    # model_name: str = "deepseek-llm:7b"
    # api_key: str = "ollama"
    # base_url: str = "http://localhost:11434/v1"

    system_prompt: str = ""
    json_mode: bool = False
    seed: int = None
    do_sample: bool = False
    temperature: float = 0
    max_tokens: int = 4096
    repetition_penalty: float = 1.05
    num_beams: int = 1
    topk: int = 50
    topp: float = 0.95

    topk_per_token: int = 5  # number of topk tokens to generate for each token

    token_usage: list = field(default_factory=list)
    request_limit: bool = True
    rpm: RPM = field(default_factory=lambda: RPM(rpm=1000))
    tpm: TPM = field(default_factory=lambda: TPM(tpm=50000))
    silent: bool = False

    def __post_init__(self):
        your_api_key = self.api_key
        if your_api_key is None:
            your_api_key = input("请输入你的API Key：")
        self.api_key = your_api_key
        self.tokenizer = Tokenizer()

    def _pre_generate(self, text: str, history: List[str]) -> Dict:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
            "max_tokens": self.max_tokens,
            "model": "openai/" + self.model_name,
            "custom_llm_provider": "",
            "api_key": self.api_key,
            "api_base": self.base_url,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})

        if history:
            # assert len(history) % 2 == 0, "History should have even number of elements."
            messages = history + messages

        kwargs['messages'] = messages
        return kwargs

    # @retry(
    #     stop=stop_after_attempt(5),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    # )
    async def generate(self, text: str, history: Optional[List[str]] = None, temperature: int = 0) -> str:
        kwargs = self._pre_generate(text, history)
        kwargs["temperature"] = temperature

        prompt_tokens = 0
        for message in kwargs['messages']:
            prompt_tokens += len(self.tokenizer.encode_string(message['content']))
        estimated_tokens = prompt_tokens + kwargs['max_tokens']
        print(f"estimated_tokens: {estimated_tokens}")

        if self.request_limit:
            await self.rpm.wait(silent=self.silent)
            await self.tpm.wait(estimated_tokens, silent=self.silent)

        # print(kwargs)

        completion = await acompletion(**kwargs)

        if hasattr(completion, "usage"):
            self.token_usage.append({
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            })
            print(self.token_usage)
        return completion.choices[0].message.content


async def main():
    async def query(sem, llm, text):
        async with sem:
            completion = await llm.generate(text)
            return completion

    client = OpenAIModel()
    semaphore = asyncio.Semaphore(3)
    task = [query(semaphore, client, q) for q in ["推理模型会给市场带来哪些新的机会", "为什么天空是蓝色的？"]]
    responses = await asyncio.gather(*task, return_exceptions=True)
    for r in responses:
        print(r)


if __name__ == '__main__':
    asyncio.run(main())
