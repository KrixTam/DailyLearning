# 下载《Operating Systems: Three Easy Pieces》的脚本

from bs4 import BeautifulSoup
import requests
import re
from dfelf import PDFFileElf
import numpy as np


def get_html_document(url):
    response = requests.get(url)
    return response.text


def download_file(url, filename):
    response = requests.get(url)
    with open(filename, mode='wb') as file:
        file.write(response.content)
    print('File <' + filename + '> is downloaded.')


if __name__ == '__main__':
    url = 'https://pages.cs.wisc.edu/~remzi/OSTEP/'
    reg = re.compile('pdf$')

    # 读取表格中的信息，并存储到列表中
    soup = BeautifulSoup(get_html_document(url), 'html.parser')
    tables = soup.find_all('table')
    t = tables[3]
    cells = t.findChildren('td')

    results = []
    for cell in cells:
        links = cell.findChildren('a')
        if len(links) > 0:
            if reg.search(links[0]['href']) is not None:
                results.append(links[0]['href'])
            else:
                results.append(None)
        else:
            results.append(None)

    # 按列整理待下载文件列表
    length = len(results)
    x = 6
    y = int(length / 6)
    filenames = np.array(results).reshape((y, x)).T.reshape((length)).tolist()
    print(filenames)
    filenames = list(filter(lambda item: item is not None, filenames))
    print(filenames)

    # 下载pdf文件
    for filename in filenames:
        download_file(url + filename, filename)
    # 拼接文件
    df_elf = PDFFileElf()
    config = {
        'input': filenames,
        'output': 'OSTEP.v1.10.pdf'
    }
    df_elf.merge(**config)
