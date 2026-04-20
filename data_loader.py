"""
数据加载和分块模块
负责加载和处理各种格式的文档(txt, pdf, docx, excel)
"""
import os
import pandas as pd
import PyPDF2
import json
import re
from docx import Document


def read_txt(file_path):
    """读取TXT文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read TXT {file_path}: {e}")
        return ""


def read_pdf(file_path):
    """读取PDF文件"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Failed to read PDF {file_path}: {e}")
    return text


def read_docx(file_path):
    """读取DOCX文件"""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Failed to read DOCX {file_path}: {e}")
        return ""


def read_excel(file_path):
    """
    读取Excel文件(.xlsx, .xls)
    策略: 读取所有工作表并将表格内容转换为结构化JSON字符串
    """
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None, dtype=str)
        full_json_list = []

        for sheet_name, df in excel_data.items():
            df.columns = df.columns.str.strip()
            # 将每一行转换为独立的JSON对象，而不是合并为一个大JSON
            records = df.to_dict(orient='records')
            for record in records:
                # 将每一行转换为JSON字符串
                json_str = json.dumps(record, ensure_ascii=False)
                full_json_list.append(json_str)

        return "\n".join(full_json_list)  # 用换行符连接，每行是一个完整的课程对象

    except Exception as e:
        print(f"Failed to read Excel {file_path}: {e}")
        return ""


def load_and_chunk_data(data_dir, chunk_size=300, chunk_overlap=50):
    """
    通用文档加载和分块函数
    核心逻辑: 优先保持JSON对象的完整性，避免将单个课程拆分
    """
    chunks = []

    # 遍历目录
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        if os.path.isdir(file_path):
            continue

        content = ""

        # 根据文件扩展名选择读取方法
        if file.endswith(".txt"):
            content = read_txt(file_path)
        elif file.endswith(".pdf"):
            content = read_pdf(file_path)
        elif file.endswith(".docx"):
            content = read_docx(file_path)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            content = read_excel(file_path)
        else:
            continue

        if not content:
            continue

        # --- 智能分块逻辑 ---

        # 1. 尝试按JSON对象分割(适用于Excel/JSON数据)
        # 假设每行或每个{...}是一个独立单元
        potential_objects = re.split(r'(?<=\})\s*(?=\{)|\n', content)

        current_chunk = ""

        for obj in potential_objects:
            obj = obj.strip()
            if not obj:
                continue

            # 跳过空值或仅标点符号
            if obj in ['{', '}', ',']:
                continue

            # 如果当前块为空，则填入
            if len(current_chunk) == 0:
                current_chunk = obj
            # 如果添加此对象不超过限制，则追加
            elif len(current_chunk) + len(obj) < chunk_size:
                current_chunk += "\n" + obj
            # 如果超过限制，则保存当前块并开始新块
            else:
                chunks.append(current_chunk)
                current_chunk = obj

        # 保存最后一个块
        if len(current_chunk) > 0:
            chunks.append(current_chunk)

    print(f"Successfully loaded and chunked: {len(chunks)} document chunks.")
    return chunks