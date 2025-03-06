# 通过指令式的提示语实现特定任务。
import ollama
from utils import timer


@timer
def demo():
    res = ollama.chat(
        model='deepseek-r1:7b',
        messages=[
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
        ],
        options={ 'temperature':0.8 }
    )

    print(res['message']['content'])


if __name__ == '__main__':
    demo()