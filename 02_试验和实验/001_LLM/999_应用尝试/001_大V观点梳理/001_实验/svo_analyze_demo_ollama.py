import ollama
import json
from utils import extract_json_from_markdown, html_to_text


class HybridSVOAnalyzer:
    def __init__(self, model="qwen3:8b"):
        # 初始化双引擎
        # self.spacy_nlp = spacy.load("zh_core_web_lg")
        self.llm_model = model
        self.confidence_threshold = 0.7

    def llm_parse(self, text):
            """Qwen增强解析"""
            try:
                prompt = self._build_prompt(text)
                print(prompt)
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options={'temperature': 0.8, 'num_ctx': 4096}
                )
                print(response)
                # 提取有效JSON部分
                json_str = extract_json_from_markdown(response['response'])
                print(json_str)
                return json.loads(json_str)
            except Exception as e:
                print(f"Qwen解析失败: {str(e)}")
                return []

    def _build_prompt(self, text):
            """动态prompt构建"""
            # json_str = '{"data": [{"subject":"张三","predicate":"批评","object":"报告"}, {"subject":"领导","predicate":"表扬","object":"李四"}]}'
            json_str = """
            {"data": [
                {"subject":"张三","predicate":"批评","object":"报告"},
                {"subject":"领导","predicate":"表扬","object":"李四"}
            ]}
            """
            return f"""
            任务：依次分析输入内容中每个句子的主谓宾结构，要求：
            1. 识别所有谓语动词及其对应的主宾
            2. 处理复杂句式（被动语态、兼语结构等）
            3. 输出JSON数组（按照句子依次输出，一个元素代表一个句子的分析结果），每个元素包含subject/predicate/object字段，
    
            示例：
            输入："张三批评了李四的报告。领导却表扬了李四。"
            输出：{json_str}
    
            当前文本：{text}
            """
            # json_str = '{"subject":"张三","predicate":"批评","object":"报告"}'
            # return f"""
            # 任务：分析句子主谓宾结构，要求：
            # 1. 识别所有谓语动词及其对应的主宾
            # 2. 处理复杂句式（被动语态、兼语结构等）
            # 3. 输出JSON数组，每个元素包含subject/predicate/object字段
            #
            # 示例：
            # 输入："张三批评了李四的报告"
            # 输出：[{json_str}]
            #
            # 当前文本：{text}
            # """


analyzer = HybridSVOAnalyzer()
# analyzer = HybridSVOAnalyzer("qwen3:14b")
# html_file = "[2025-06-12]又吃大肉明天这样做.html"
html_file = "/02_试验和实验/001_LLM/999_应用尝试/001_大V观点梳理/001_实验/[2025-06-12]又吃大肉明天这样做.html"
# content = html_to_text(html_file).split('精选留言')[0]
content = html_to_text(html_file)
# content = '虽然天气恶劣，但工程队仍然按时完成了桥梁建设。张老师等三位老师今天来上课。'
result = analyzer.llm_parse(content)
print(result)
