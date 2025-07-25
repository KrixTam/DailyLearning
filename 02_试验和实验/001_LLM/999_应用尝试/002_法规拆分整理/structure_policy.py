import json
import asyncio
from llm.utils import extract_json_from_markdown, to_md, split_pdf
from llm.openai_model import OpenAIModel
from string import Template
from llm.loop import create_event_loop
from moment import moment

TEMPLATE = Template("""任务：从给定文本中拆分整理法规条文，要求：
1. 自高往低按照编、章、条、款供4个级别整理法规条文，如果不存在某个级别的情况，该级别的内容为""；
2. 输出JSON数组，每个元素包含编、章、条、款、content共5个字段，其中：content为原文的引用内容；
3. 不能把示例的输出作为输出内容整合到最后的结果中。
示例：
    输入：
        证券期货业网络和信息安全管理办法
        （ 2023 年 月 1 17 日中国证券监督管理委员会第 次委务会议 1 审议通过）
        第一章 总 则
        第一条 为了保障证券期货业网络和信息安全，保护投资者 合法权益，促进证券期货业稳定健康发展。
        第二章 网络和信息安全运行
        第九条 信息技术系统服务机构应当建立网络和信息安全管理制度， 配备相应的安全、合规管理人员，建立与提供产品或者服务相适 应的网络和信息安全管理机制。     
        第十条 核心机构和经营机构应当建立网络和信息安全工作协调和决 策机制，保障第一责任人和直接责任人履行职责。                          
    输出：
        $json_str
当前文本：
    $content
""")
DEFAULT_ANALYSIS_RESULT = '{"data": []}'


class PolicyAnalyzer:
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

    def analyze(self, pdf_filename: str, interval: int, max_concurrent: int, artifacts_path_local: str = None, as_completed: bool = True):
        loop = create_event_loop()
        loop.run_until_complete(async_analyze(
            self, pdf_filename, interval, max_concurrent, artifacts_path_local, as_completed
        ))

    async def parse(self, text):
        """LLM增强解析"""
        def build_prompt(content: str):
            """动态prompt构建"""
            json_str = '[{"编": "", "章": "第一章", "条": "第一条", "款": "", "content": "第一条 为了保障证券期货业网络和信息安全，保护投资者 合法权益，促进证券期货业稳定健康发展。"}, {"编": "", "章": "第二章", "条": "第九条", "款": "", "content": "第九条 信息技术系统服务机构应当建立网络和信息安全管理制度， 配备相应的安全、合规管理人员，建立与提供产品或者服务相适 应的网络和信息安全管理机制。"}, {"编": "", "章": "第二章", "条": "第十条", "款": "", "content": "第十条 核心机构和经营机构应当建立网络和信息安全工作协调和决 策机制，保障第一责任人和直接责任人履行职责。"}]'
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


async def async_analyze(llm: PolicyAnalyzer, pdf_filename: str, interval: int, max_concurrent: int, artifacts_path_local: str = None, as_completed: bool = True):
    async def get_policy(sem: asyncio.Semaphore, llm: PolicyAnalyzer, md_content: str):
        async with sem:
            try:
                result = await llm.parse(md_content)
            except Exception as e:
                print(f"LLM解析失败(async_get_policy): {str(e)}")
                result = None
            return result

    semaphore = asyncio.Semaphore(max_concurrent)
    pdf_files = split_pdf(pdf_filename, interval)
    md_file_contents = []
    for pdf_file in pdf_files:
        md_file_content = to_md(pdf_file, artifacts_path_local)
        md_file_contents.append(md_file_content)
    tasks = [get_policy(semaphore, llm, md_content) for md_content in md_file_contents]
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


async def main(llm: PolicyAnalyzer, pdf_filename: str, artifacts_path_local: str = None):
    print(moment().format())
    md_file_content = to_md(pdf_filename, artifacts_path_local)
    response = await llm.parse(md_file_content)
    print(response)
    print(moment().format())


if __name__ == '__main__':
    base_dir = "/Users/krix/PycharmProjects/DailyLearning/02_试验和实验/001_LLM/999_应用尝试/002_法规拆分整理/"
    small_pdf_file = base_dir + "中华人民共和国民法典_test_small.pdf"
    pdf_file = base_dir + "中华人民共和国民法典_test_1.pdf"
    artifacts_path_local = "/Users/krix/.cache/docling/models"
    policy_analyzer = PolicyAnalyzer()
    policy_analyzer.set_temperature(0.2)
    test_case = 2
    if test_case == 1:
        # case 01
        asyncio.run(main(policy_analyzer, small_pdf_file, artifacts_path_local))
    else:
        # case 02
        policy_analyzer.analyze(pdf_file, 3, 6, artifacts_path_local, False)
