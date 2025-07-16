import json
from openai import OpenAI
from utils import extract_json_from_markdown, extract_html_info, to_md


class IntentionAnalyzer:
    def __init__(self, client=None, model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        # 初始化双引擎
        self.llm_model = model
        if client is None:
            your_api_key = input("请输入你的API Key：")
            self.client = OpenAI(api_key=your_api_key, base_url="https://api.siliconflow.cn/v1")
        else:
            self.client = client

    def analyze(self, html_filename, encoding='utf-8', artifacts_path_local: str = None):
        result = extract_html_info(html_filename, encoding)
        md_file = to_md(html_filename, artifacts_path_local)
        with open(md_file, 'r') as f:
            content = f.read()
            result['intention'] = self.parse(content)
            return result

    def parse(self, text):
        """LLM增强解析"""
        def build_prompt(content: str):
            """动态prompt构建"""
            json_str = '[{"condition": "", "intent": "继续持有", "stock": ["顺钠科技", "中恒电气", "海得控制"], "content": "昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行"}]'
            return f"""
            任务：从给定文本中提取出疑似股票交易意图信息，要求：
            1. 识别所有交易意图（类似买入、卖出、持有、观望，根据内容进行归纳总结描述意图）以及对应的股票名称；
            2. 输出JSON数组，每个元素包含condition、intent、stock、content四个字段，其中：content为原文的引用内容；stock字段是一个列表；condition字段需要从交易意图中识别潜在的限制条件，比如未来的走势变化、指标值达到某一限定值等等，如果没有明确条件，就把condition设置为""就好了；intent为总结归纳的交易意图；
            3. 不能把示例的输出作为输出内容整合到最后的结果中。

            示例：
            输入："昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行，别太离谱明天还可能会有人救。"
            输出：{json_str}

            当前文本：{content}
            """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        'role': 'user',
                        'content': build_prompt(text)
                    }
                ],
                temperature=0.2,
                stream=False
            )
            response_text = '{"data": ' + response.choices[0].message.content + '}'
            # print("S1: " + response_text)
            # 提取有效JSON部分
            json_response = extract_json_from_markdown(response_text)
            # print("S2: " + json_str)
            return json.loads(json_response)
        except Exception as e:
            print(f"LLM解析失败: {str(e)}")
            return []


if __name__ == '__main__':
    intention_analyzer = IntentionAnalyzer()
    html_file = "/Users/krix/PycharmProjects/DailyLearning/02_试验和实验/001_LLM/999_应用尝试/001_大V观点梳理/001_实验/[2025-06-12]又吃大肉明天这样做.html"
    artifacts_path_local = "/Users/krix/.cache/docling/models"
    response = intention_analyzer.analyze(html_file, artifacts_path_local=artifacts_path_local)
    print(response)
