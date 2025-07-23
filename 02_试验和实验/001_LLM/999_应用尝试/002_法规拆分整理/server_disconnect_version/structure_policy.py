import json
import asyncio
from dataclasses import dataclass
from utils import to_md, extract_json_from_markdown, split_pdf
from openai_model import OpenAIModel
from loop import create_event_loop


@dataclass
class PolicyAnalyzer:
    max_concurrent = 2

    def __post_init__(self):
        self.client = OpenAIModel()

    async def parse(self, semaphore, md_file):
        """LLM解析"""
        def build_prompt(content: str):
            """动态prompt构建"""
            json_str = '{"data": [{"编": "", "章": "第一章", "条": "第一条", "款": "", "content": "第一条 为了保障证券期货业网络和信息安全，保护投资者 合法权益，促进证券期货业稳定健康发展。"}, {"编": "", "章": "第二章", "条": "第九条", "款": "", "content": "第九条 信息技术系统服务机构应当建立网络和信息安全管理制度， 配备相应的安全、合规管理人员，建立与提供产品或者服务相适 应的网络和信息安全管理机制。"}, {"编": "", "章": "第二章", "条": "第十条", "款": "", "content": "第十条 核心机构和经营机构应当建立网络和信息安全工作协调和决 策机制，保障第一责任人和直接责任人履行职责。"}]}'
            return f"""
任务：从给定文本中拆分整理法规条文，要求：
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
        {json_str}
当前文本：
    {content}
"""

        def build_prompt_02(content: str):
            """动态prompt构建"""
            json_str = '{"data": ["第一章 总 则", "第一条 为了保障证券期货业网络和信息安全，保护投资者 合法权益，促进证券期货业稳定健康发展。", "第二章 网络和信息安全运行", "第九条 信息技术系统服务机构应当建立网络和信息安全管理制度， 配备相应的安全、合规管理人员，建立与提供产品或者服务相适 应的网络和信息安全管理机制。", "第十条 核心机构和经营机构应当建立网络和信息安全工作协调和决 策机制，保障第一责任人和直接责任人履行职责。"]}'
            return f"""
任务：从给定文本中拆分整理法规条文，要求：
1. 按照编、章、条、款供4个维度整理法规条文；
2. 输出JSON数组，每个元素为原文对应编、章、条、款的内容；
3. 不能把示例的输出作为输出内容整合到最后的结果中。
示例：
输入：证券期货业网络和信息安全管理办法\n（ 2023 年 月 1 17 日中国证券监督管理委员会第 次委务会议 1 审议通过）\n第一章 总 则\n第一条 为了保障证券期货业网络和信息安全，保护投资者 合法权益，促进证券期货业稳定健康发展。\n第二章 网络和信息安全运行\n第九条 信息技术系统服务机构应当建立网络和信息安全管理制度， 配备相应的安全、合规管理人员，建立与提供产品或者服务相适 应的网络和信息安全管理机制。\n第十条 核心机构和经营机构应当建立网络和信息安全工作协调和决 策机制，保障第一责任人和直接责任人履行职责。
输出：{json_str}

当前文本：{content}
"""

        async with semaphore:
            f = open(md_file, 'r')
            text = f.read()
            print("Sending Query...")
            response_text = await self.client.generate(build_prompt(text))
            try:
                # print("S1: " + response_text)
                # 提取有效JSON部分
                json_response = extract_json_from_markdown(response_text)
                # print("S2: " + json_response)
                return json.loads(json_response)
            except Exception as e:
                print(f"LLM解析失败: {str(e)} \n\n ================= \n\n {response_text}")
                return {}


def analyze(llm_client: PolicyAnalyzer, pdf_filename, interval: int = 10, artifacts_path_local: str = None):
    loop = create_event_loop()
    loop.run_until_complete(async_analyze(llm_client, llm_client.max_concurrent, pdf_filename, interval, artifacts_path_local))


async def async_analyze(llm_client: PolicyAnalyzer, max_concurrent, pdf_filename, interval: int = 10, artifacts_path_local: str = None):
    semaphore = asyncio.Semaphore(max_concurrent)
    pdf_files = split_pdf(pdf_filename, interval)
    md_files = []
    for pdf_file in pdf_files:
        md_file = to_md(pdf_file, artifacts_path_local)
        md_files.append(md_file)
    calls = [llm_client.parse(semaphore, md_file) for md_file in md_files]
    # 方案一
    # responses = await asyncio.gather(*calls, return_exceptions=True)
    # for r in responses:
    #     print(r)
    # 方案二
    for completed_task in asyncio.as_completed(calls):
        response = await completed_task
        print(response)


if __name__ == '__main__':
    policy_analyzer = PolicyAnalyzer()
    artifacts_path_local = "/Users/krix/.cache/docling/models"
    pdf_filename = '../中华人民共和国民法典_test_1.pdf'
    analyze(policy_analyzer, pdf_filename, 3, artifacts_path_local)
