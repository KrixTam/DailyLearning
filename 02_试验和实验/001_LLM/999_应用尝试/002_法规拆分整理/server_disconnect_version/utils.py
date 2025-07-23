import os
import re
import json
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from PyPDF2 import PdfReader, PdfWriter
import logging


logger = logging.getLogger("DailyLearning")


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
        print(f"JSON解析失败: {str(e)}\n\n ================= \n\n {json_str}")
        json_str = '{"data": ' + json_str + "}"
        return json_str


def to_md(input_filename, artifacts_path: str = None):
    """
    PDF转Markdown
    """
    if artifacts_path is None:
        converter = DocumentConverter()
    else:
        # 可以使用命令行先下载模型
        # docling-tools models download
        pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    filename = os.path.splitext(os.path.basename(input_filename))[0]
    result = converter.convert(input_filename)
    markdown_text = result.document.export_to_markdown()
    output_dir = mkdir_output()
    # 保存到本地 Markdown 文件
    output_filename = os.path.join(output_dir, filename + ".md")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"文件「{input_filename}」已转换为Markdown格式，并保存到：{output_filename}")
    return output_filename


def mkdir_output():
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def split_pdf(pdf_filename, interval: int = 20):
    with open(pdf_filename, 'rb') as file:
        pdf_reader = PdfReader(file)
        total_pages = len(pdf_reader.pages)
        if total_pages <= interval + 3:
            return [pdf_filename]
        count = 0
        index = 1
        pdf_writer = PdfWriter()
        pdf_writer_next = None
        output_dir = mkdir_output()
        filename = output_dir + "/" + os.path.splitext(os.path.basename(pdf_filename))[0]
        output_files = []
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
            count = count + 1
            if count == interval:
                pdf_writer_next = PdfWriter()
            if pdf_writer_next is not None:
                pdf_writer_next.add_page(page)
            if count > interval:
                output_filename = filename + "_" + str(index) + ".pdf"
                with open(output_filename, 'wb') as out:
                    pdf_writer.write(out)
                    output_files.append(output_filename)
                    index = index + 1
                    count = 1
                    pdf_writer = pdf_writer_next
                    pdf_writer_next = None
        if pdf_writer is not None:
            output_filename = filename + "_" + str(index) + ".pdf"
            with open(output_filename, 'wb') as out:
                pdf_writer.write(out)
                output_files.append(output_filename)
        return output_files
