import jieba.posseg as pseg
import re
from collections import defaultdict


class SVOAnalyzer:
    def __init__(self):
        self.punctuations = set("，。！？；：、")
        self.auxiliary_words = {"了", "着", "过", "的", "地", "得"}
        self.subject_tags = {'n', 'r', 'nr', 'ns'}
        self.predicate_tags = {'v', 'vd', 'vn'}
        self.object_tags = {'n', 't', 'r', 'nr'}

    def analyze(self, text):
        sentences = re.split(r'[。！？]', text)
        results = []

        for sent in sentences:
            if not sent.strip():
                continue
            words = list(pseg.cut(sent))  # 转换为列表
            svos = self._parse_sentence(words)
            results.extend(svos)

        return {"data": results}

    def _parse_sentence(self, words):
        svos = []
        subject = None
        predicate = None
        object_ = []
        n = len(words)

        for i in range(n):
            word, flag = words[i]  # 正确解包词和词性

            # 主语识别（名词/代词且在动词前）
            if self._is_subject_candidate(i, words):
                subject = word
                continue

            # 谓语识别（动词）
            if self._is_predicate(i, words):
                predicate = word
                continue

            # 宾语识别（动词后的名词短语）
            if predicate and self._is_object_candidate(i, words):
                object_.append(word)
                continue

            # 遇到句末标点或助词结束当前分析
            if word in self.punctuations or word in self.auxiliary_words:
                if subject and predicate and object_:
                    svos.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": "".join(object_)
                    })
                    # 重置状态
                    subject = None
                    predicate = None
                    object_ = []

        return svos

    def _is_subject_candidate(self, index, words):
        """判断是否为主语候选（名词/代词且在动词前）"""
        if index >= len(words) - 1:
            return False
        # 直接获取当前词性
        current_flag = words[index].flag
        # 检查后续是否有动词
        for i in range(index + 1, len(words)):
            if words[i].flag.startswith('v'):
                return current_flag in self.subject_tags
        return False

    def _is_predicate(self, index, words):
        """判断是否为谓语动词"""
        current_flag = words[index].flag
        return current_flag in self.predicate_tags and words[index].word not in {"是", "有", "在"}

    def _is_object_candidate(self, index, words):
        """判断是否为宾语候选（名词短语）"""
        return words[index].flag in self.object_tags


# 测试用例
if __name__ == "__main__":
    analyzer = SVOAnalyzer()
    text = "张三批评了李四的报告。领导却表扬了李四。"
    result = analyzer.analyze(text)
    print(result)
