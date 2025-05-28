# 通过指令式的提示语实现特定任务。
from vllm import LLM, SamplingParams
from utils import timer, print_outputs
import vllm.envs as envs
envs.VLLM_HOST_IP = "0.0.0.0"
envs.VLLM_CPU_KVCACHE_SPACE = 40


@timer
def demo():
    params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
    # 需要提前把模型下载到本地
    # git lfs clone https://huggingface.co/Qwen/Qwen3-4B
    # 其他参考：
    # https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
    llm = LLM(model='/Users/krix/vllm/models/Qwen3-4B')

    messages = [
        {
            'role': 'system',
            'content': '你是一位数据分析师，请根据以下步骤撰写一份数据分析报告。'
        },
        {
            'role': 'user',
            'content': '''背景：电商平台要分析用户购物行为。请完成以下任务：
            1. 提出2~3个切入点；
            2. 列出每个切入点要分析的指标；
            3. 假设你发现了有价值的洞见，提出2~3条可行的建议；
            4. 综合成一份完整报告。
            '''
        }
    ]

    res = llm.chat(
        messages=[messages],
        sampling_params=params,
        chat_template_kwargs={"enable_thinking": True},
        use_tqdm=False
    )

    print_outputs(res)


if __name__ == '__main__':
    demo()
