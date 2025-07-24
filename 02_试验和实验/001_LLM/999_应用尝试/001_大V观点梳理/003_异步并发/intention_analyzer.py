import json
import asyncio
from llm.utils import extract_json_from_markdown, extract_html_info, to_md, logger
from llm.openai_model import OpenAIModel
from string import Template
from llm.loop import create_event_loop
from moment import moment

TEMPLATE = Template("""任务：从给定文本中提取出疑似股票交易意图信息，要求：
1. 识别所有交易意图（类似买入、卖出、持有、观望，根据内容进行归纳总结描述意图）以及对应的股票名称；
2. 输出JSON数组，每个元素包含condition、intent、stock、content四个字段，其中：content为原文的引用内容；stock字段是一个列表；condition字段需要从交易意图中识别潜在的限制条件，比如未来的走势变化、指标值达到某一限定值等等，如果没有明确条件，就把condition设置为""就好了；intent为总结归纳的交易意图；
3. 不能把示例的输出作为输出内容整合到最后的结果中。

示例：
输入："昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行，别太离谱明天还可能会有人救。"
输出：$json_str

当前文本：$content""")
DEFAULT_ANALYSIS_RESULT = '{"data": []}'


class IntentionAnalyzer:
    def __init__(self, client: OpenAIModel = None, model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        if client is None:
            self.llm = OpenAIModel()
        else:
            self.llm = client
        self.llm.model_name = model
        self.set_temperature(0.2)

    def get_temperature(self):
        return self.llm.temperature

    def set_temperature(self, temperature: float):
        if temperature >= 0:
            self.llm.temperature = temperature

    def analyze(self, html_files: list, max_concurrent: int, encoding: str = 'utf-8', artifacts_path_local: str = None, as_completed: bool = True):
        loop = create_event_loop()
        loop.run_until_complete(async_analyze(
            self, html_files, max_concurrent, encoding, artifacts_path_local, as_completed
        ))

    async def async_get_intention(self, html_filename: str, encoding: str = 'utf-8', artifacts_path_local: str = None):
        result = extract_html_info(html_filename, encoding)
        # 使用Markdown内容进行分析
        # md_file_content = to_md(html_filename, artifacts_path_local)
        # result['intention'] = await self.parse(md_file_content)
        # 使用纯文本内容进行分析
        result["intention"] = await self.parse(result["content"])
        return result

    async def parse(self, text):
        """LLM增强解析"""
        def build_prompt(content: str):
            """动态prompt构建"""
            json_str = '[{"condition": "", "intent": "继续持有", "stock": ["顺钠科技", "中恒电气", "海得控制"], "content": "昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行"}]'
            return TEMPLATE.substitute(json_str=json_str, content=content)
        try:
            response_text = await self.llm.generate(build_prompt(text), temperature=self.get_temperature())
            response_text = '{"data": ' + response_text + '}'
            # print("S1: " + response_text)
            # 提取有效JSON部分
            json_response = extract_json_from_markdown(response_text)
            # print("S2: " + json_str)
            return json.loads(json_response)
        except Exception as e:
            print(f"LLM解析失败: {str(e)}")
            return DEFAULT_ANALYSIS_RESULT


async def async_analyze(llm: IntentionAnalyzer, html_files: list, max_concurrent: int, encoding: str = 'utf-8', artifacts_path_local: str = None, as_completed: bool = True):
    async def get_intention(sem, llm, html_file: str, encoding: str = 'utf-8', artifacts_path_local: str = None):
        async with sem:
            try:
                result = await llm.async_get_intention(html_file, encoding, artifacts_path_local)
            except Exception as e:
                print(f"LLM解析失败: {str(e)}")
                result = None
            return result

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [get_intention(semaphore, llm, html_file, encoding, artifacts_path_local) for html_file in html_files]
    print(moment().format())
    if as_completed:
        # case 02_02
        for completed_task in asyncio.as_completed(tasks):
            response = await completed_task
            print(response)
    else:
        # case 02_01
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for r in responses:
            print(r)
    print(moment().format())


async def main(llm: IntentionAnalyzer, html_file: str, artifacts_path_local: str):
    print(moment().format())
    response = await llm.async_get_intention(html_file, artifacts_path_local=artifacts_path_local)
    print(response)
    print(moment().format())


if __name__ == '__main__':
    base_dir = "/Users/krix/PycharmProjects/DailyLearning/02_试验和实验/001_LLM/999_应用尝试/001_大V观点梳理/001_实验/"
    html_files = [base_dir + '[2025-03-16]注意年报雷.html', base_dir + '[2025-06-12]又吃大肉明天这样做.html']
    artifacts_path_local = "/Users/krix/.cache/docling/models"
    intention_analyzer = IntentionAnalyzer()
    intention_analyzer.set_temperature(0.525)
    test_case = 2
    if test_case == 1:
        # case 01
        asyncio.run(main(intention_analyzer, html_files[1], artifacts_path_local))
    else:
        # case 02
        intention_analyzer.analyze(
            html_files, 10, artifacts_path_local=artifacts_path_local, as_completed=True
        )
