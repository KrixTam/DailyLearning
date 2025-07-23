import tiktoken
from dataclasses import dataclass
from typing import List


try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


def get_tokenizer(tokenizer_name: str = "cl100k_base"):
    """
    Get a tokenizer instance by name.

    :param tokenizer_name: tokenizer name, tiktoken encoding name or Hugging Face model name
    :return: tokenizer instance
    """
    if tokenizer_name in tiktoken.list_encoding_names():
        return tiktoken.get_encoding(tokenizer_name)
    if TRANSFORMERS_AVAILABLE:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from Hugging Face: {e}") from e
    else:
        raise ValueError("Hugging Face Transformers is not available, please install it first.")


@dataclass
class Tokenizer:
    model_name: str = "cl100k_base"

    def __post_init__(self):
        self.tokenizer = get_tokenizer(self.model_name)

    def encode_string(self, text: str) -> List[int]:
        """
        Encode text to tokens

        :param text
        :return: tokens
        """
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens to text

        :param tokens
        :return: text
        """
        return self.tokenizer.decode(tokens)

    def chunk_by_token_size(
        self, content: str, overlap_token_size=128, max_token_size=1024
    ):
        tokens = self.encode_string(content)
        results = []
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = self.decode_tokens(
                tokens[start: start + max_token_size]
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
        return results


if __name__ == '__main__':
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
    c = """## 中华人民共和国民法典

中华人民共和国主席令第四十五号

《中华人民共和国民法典》已由中华人民共和国第十三届全国人民代表大会第三次 会议于 2020 年 5 月 28 日通过，现予公布，自 2021 年 1 月 1 日起施行。

中华人民共和国主席 习近平

2020 年 5 月 28 日

（ 2020 年 5 月 28 日第十三届全国人民代表大会第三次会议通过）

第一编 总则

第一章 基本规定

## 第一条

为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特 色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

## 第二条

民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

## 第三条

民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人 不得侵犯。

## 第四条

民事主体在民事活动中的法律地位一律平等。

## 第五条

民事主体从事民事活动，应当遵循自愿原则，按照自己的意思设立、变更、终止民 事法律关系。

## 第六条

民事主体从事民事活动，应当遵循公平原则，合理确定各方的权利和义务。

## 第七条

民事主体从事民事活动，应当遵循诚信原则，秉持诚实，恪守承诺。

## 第八条

民事主体从事民事活动，不得违反法律，不得违背公序良俗。

## 第九条

民事主体从事民事活动，应当有利于节约资源、保护生态环境。

## 第十条

处理民事纠纷，应当依照法律；法律没有规定的，可以适用习惯，但是不得违背公 序良俗。

## 第十一条

其他法律对民事关系有特别规定的，依照其规定。

## 第十二条

中华人民共和国领域内的民事活动，适用中华人民共和国法律。法律另有规定的， 依照其规定。

## 第二章 自然人

第一节 民事权利能力和民事行为能力

## 第十三条

自然人从出生时起到死亡时止，具有民事权利能力，依法享有民事权利，承担民事 义务。

## 第十四条

自然人的民事权利能力一律平等。

## 第十五条

自然人的出生时间和死亡时间，以出生证明、死亡证明记载的时间为准；没有出生 证明、死亡证明的，以户籍登记或者其他有效身份登记记载的时间为准。有其他证 据足以推翻以上记载时间的，以该证据证明的时间为准。

## 第十六条

涉及遗产继承、接受赠与等胎儿利益保护的，胎儿视为具有民事权利能力。但是， 胎儿娩出时为死体的，其民事权利能力自始不存在。

## 第十七条

十八周岁以上的自然人为成年人。不满十八周岁的自然人为未成年人。

## 第十八条

成年人为完全民事行为能力人，可以独立实施民事法律行为。

十六周岁以上的未成年人，以自己的劳动收入为主要生活来源的，视为完全民事行 为能力人。

## 第十九条

八周岁以上的未成年人为限制民事行为能力人，实施民事法律行为由其法定代理人

代理或者经其法定代理人同意、追认；但是，可以独立实施纯获利益的民事法律行 为或者与其年龄、智力相适应的民事法律行为。

## 第二十条

不满八周岁的未成年人为无民事行为能力人，由其法定代理人代理实施民事法律行 为。

## 第二十一条

不能辨认自己行为的成年人为无民事行为能力人，由其法定代理人代理实施民事法 律行为。

八周岁以上的未成年人不能辨认自己行为的，适用前款规定。

## 第二十二条

不能完全辨认自己行为的成年人为限制民事行为能力人，实施民事法律行为由其法 定代理人代理或者经其法定代理人同意、追认；但是，可以独立实施纯获利益的民 事法律行为或者与其智力、精神健康状况相适应的民事法律行为。

## 第二十三条

无民事行为能力人、限制民事行为能力人的监护人是其法定代理人。

## 第二十四条

不能辨认或者不能完全辨认自己行为的成年人，其利害关系人或者有关组织，可以 向人民法院申请认定该成年人为无民事行为能力人或者限制民事行为能力人。

被人民法院认定为无民事行为能力人或者限制民事行为能力人的，经本人、利害关 系人或者有关组织申请，人民法院可以根据其智力、精神健康恢复的状况，认定该 成年人恢复为限制民事行为能力人或者完全民事行为能力人。

本条规定的有关组织包括：居民委员会、村民委员会、学校、医疗机构、妇女联合"""
    t = Tokenizer()
    print(t.chunk_by_token_size(build_prompt(c)))
