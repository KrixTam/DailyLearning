import re
import json
from bs4 import BeautifulSoup


def extract_json_from_markdown(text):
    # 正则模式：匹配```json块并捕获内容（支持多行）
    pattern = re.compile(r'```json(.*?)```', re.DOTALL)
    # 查找所有匹配项
    matches = pattern.findall(text)
    if len(matches) > 0:
        # 清理每个匹配项的首尾空白并返回
        # return [match.strip() for match in matches]
        json_str = matches[0].strip()
    else:
        json_str_list = text.split('</think>\n\n')
        if len(json_str_list) == 1:
            json_str = json_str_list[0]
        else:
            json_str = json_str_list[1]
    try:
        a = json.loads(json_str)
        if isinstance(a['data'], list):
            return json_str
    except Exception as e:
        print(f"解析失败: {str(e)}")
        json_str = '{"data": ' + json_str + "}"
        return json_str


def html_to_text(html_path, encoding='utf-8'):
    """
    从HTML文件中提取纯文本并导出到文本文件

    参数:
        html_path (str): 输入HTML文件路径
        output_path (str): 输出文本文件路径（如："output.txt"）
        encoding (str): 文件编码（默认utf-8）
    """
    # 读取HTML文件内容
    with open(html_path, 'r', encoding=encoding) as f:
        html_content = f.read()

    # 解析HTML（使用lxml解析器，需安装；若没有可用html.parser）
    soup = BeautifulSoup(html_content, 'lxml')  # 或 'html.parser'
    # 移除不需要的标签（如script、style、noscript等）
    for tag in soup(['script', 'style', 'noscript', 'svg']):
        tag.decompose()  # 彻底删除标签及其内容

    element = soup.find(id='js_content')

    text = element.get_text(separator='\n', strip=True)
    # 进一步清理多余空白（如连续换行）
    cleaned_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    # 进一步清理多余换行
    cleaned_text = ''.join(cleaned_text.split('\n'))
    return cleaned_text
