import spacy
import json
from openai import OpenAI
from utils import extract_json_from_markdown, html_to_text
from functools import lru_cache


your_api_key = input("请输入你的API Key：")
client = OpenAI(api_key=your_api_key,
                base_url="https://api.siliconflow.cn/v1")


class HybridSVOAnalyzer:
    def __init__(self, model="Qwen/Qwen3-8B"):
        # 初始化双引擎
        # self.spacy_nlp = spacy.load("zh_core_web_lg")
        self.llm_model = model
        self.confidence_threshold = 0.7
        # 初始化缓存
        # self._cache_spacy_parse = lru_cache(maxsize=1000)(self._spacy_parse)

    def llm_parse(self, text):
            """Qwen增强解析"""
            try:
                response = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            'role': 'user',
                            'content': self._build_prompt(text)
                        }
                    ],
                    temperature=0.2,
                    stream=False
                )
                response_text = response.choices[0].message.content
                print("S1: " + response_text)
                # 提取有效JSON部分
                json_str = extract_json_from_markdown(response_text)
                print("S2: " + json_str)
                return json.loads(json_str)
            except Exception as e:
                print(f"Qwen解析失败: {str(e)}")
                return []

    def _build_prompt(self, text):
        # return self._build_prompt_01(text)
        # return self._build_prompt_02(text)
        # return self._build_prompt_03(text)
        # return self._build_prompt_04(text)
        return self._build_prompt_05(text)
        # return self._build_prompt_06(text)

    def _build_prompt_01(self, text):
        """动态prompt构建"""
        json_str = '{"data": [{"subject":"张三","predicate":"批评","object":"报告"}, {"subject":"领导","predicate":"表扬","object":"李四"}]}'
        return f"""
        任务：依次分析输入内容中每个句子的主谓宾结构，要求：
        1. 识别所有谓语动词及其对应的主宾；
        2. 处理复杂句式（被动语态、兼语结构等）；
        3. 输出JSON数组（按照句子依次输出，一个元素代表一个句子的分析结果），每个元素包含subject/predicate/object字段。

        示例：
        输入："张三批评了李四的报告。领导却表扬了李四。"
        输出：{json_str}

        当前文本：{text}
        """
    def _build_prompt_02(self, text):
        """动态prompt构建"""
        json_str = '{"data": [{"noun": ["张三", "报告"], "verb": "批评"},{"noun": ["领导", "李四"], "verb": "表扬"}]}'
        return f"""
        任务：依次分析输入内容中每个句子的动词和名词，要求：
        1. 识别所有动词和名词
        2. 处理复杂句式（被动语态、兼语结构等）
        3. 输出JSON数组（按照句子依次输出，一个元素代表一个句子的分析结果），每个元素包含verb/noun字段，

        示例：
        输入："张三批评了李四的报告。领导却表扬了李四。"
        输出：{json_str}

        当前文本：{text}
        """

    def _build_prompt_03(self, text):
        """动态prompt构建"""
        json_str = '{"data": [{"action": "持有", "stock": ["顺钠科技", "中恒电气", "海得控制"]}]}'
        return f"""
        任务：从给定文本中提取出疑似股票交易的信息，要求：
        1. 识别所有交易动作（类似买入、卖出、持有、观望）以及对应的股票名称；
        2. 输出JSON数组，每个元素包含action、stock字段，stock字段是一个列表；
        3. 不能把示例的输出作为输出内容整合到最后的结果中。

        示例：
        输入："昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行，别太离谱明天还可能会有人救。"
        输出：{json_str}

        当前文本：{text}
        """

    def _build_prompt_04(self, text):
        """动态prompt构建"""
        json_str = '{"data": [{"intent": "继续持有", "stock": ["顺钠科技", "中恒电气", "海得控制"]}]}'
        return f"""
        任务：从给定文本中提取出疑似股票交易意图信息，要求：
        1. 识别所有交易意图（类似买入、卖出、持有、观望，根据内容进行归纳总结描述意图）以及对应的股票名称；
        2. 输出JSON数组，每个元素包含intent、stock字段，stock字段是一个列表；
        3. 不能把示例的输出作为输出内容整合到最后的结果中。

        示例：
        输入："昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行，别太离谱明天还可能会有人救。"
        输出：{json_str}

        当前文本：{text}
        """

    def _build_prompt_05(self, text):
        """动态prompt构建"""
        json_str = '{"data": [{"condition": "", "intent": "继续持有", "stock": ["顺钠科技", "中恒电气", "海得控制"]}]}'
        return f"""
        任务：从给定文本中提取出疑似股票交易意图信息，要求：
        1. 识别所有交易意图（类似买入、卖出、持有、观望，根据内容进行归纳总结描述意图）以及对应的股票名称；
        2. 输出JSON数组，每个元素包含condition、intent、stock字段，stock字段是一个列表；其中condition字段需要从交易意图中识别潜在的限制条件，比如未来的走势变化、指标值达到某一限定值等等，如果没有明确条件，就把condition设置为""就好了；
        3. 不能把示例的输出作为输出内容整合到最后的结果中。

        示例：
        输入："昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行，别太离谱明天还可能会有人救。"
        输出：{json_str}

        当前文本：{text}
        """

    def _build_prompt_06(self, text):
        """动态prompt构建"""
        json_str = '{"trading": [{"intent": "继续持有", "stock": ["顺钠科技", "中恒电气", "海得控制"]}], "intention": [{"condition": "", "intent": "继续持有", "stock": ["顺钠科技", "中恒电气", "海得控制"]}]}'
        return f"""
        读取提供的内容，并按照如下要求完成任务1和任务2，并按照示例输出JSON处理的结果。
        
        任务1：从给定文本中提取出疑似股票交易意图信息，要求：
        1. 识别所有交易意图（类似买入、卖出、持有、观望，根据内容进行归纳总结描述意图）以及对应的股票名称；
        2. 以JSON格式输出处理结果放到trading中，内容为一个数组，每个元素包含intent、stock字段，stock字段是一个列表；
        
        任务2：从给定文本中提取出疑似股票交易意图信息，要求：
        1. 识别所有交易意图（类似买入、卖出、持有、观望，根据内容进行归纳总结描述意图）以及对应的股票名称；
        2. 以JSON格式输出处理结果放到intention中，内容为一个数组，每个元素包含condition、intent、stock字段，stock字段是一个列表；其中condition字段需要从交易意图中识别潜在的限制条件，比如未来的走势变化、指标值达到某一限定值等等，如果没有明确条件，就把condition设置为""就好了；
        
        其他要求：
        1. 不能把示例的输出作为输出内容整合到最后的结果中；
        2. trading和intention的内容不能自相矛盾。

        示例：
        输入："昨天的三个票今天板了一个顺钠科技，中恒电气还行，海得控制也还行，至少目前都还行，别太离谱明天还可能会有人救。"
        输出：{json_str}
        
        待分析处理的内容如下，请读取后并按照以上要求输出处理结果（注意，禁止将示例的输出内容作为最后处理结果的一部分内容进行输出）：
        {text}
        """


analyzer = HybridSVOAnalyzer("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
# analyzer = HybridSVOAnalyzer()
# html_file = "/Users/krix/PycharmProjects/DailyLearning/02_试验和实验/002_QS/001_大V观点梳理/001_实验/[2025-06-12]又吃大肉明天这样做.html"
html_file = "/Users/krix/PycharmProjects/DailyLearning/02_试验和实验/002_QS/001_大V观点梳理/001_实验/往伤口里倒白酒.html"
# html_file = "/Users/krix/PycharmProjects/DailyLearning/02_试验和实验/002_QS/001_大V观点梳理/001_实验/[2025-03-16]注意年报雷.html"
content = html_to_text(html_file)
# content = '虽然天气恶劣，但工程队仍然按时完成了桥梁建设。张老师等三位老师今天来上课。'
result = analyzer.llm_parse(content)
print(result)
