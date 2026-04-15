import os
import PyPDF2
from docx import Document
import pandas as pd  # 新增：用于处理 Excel


def read_txt(file_path):
    """读取 TXT 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_pdf(file_path):
    """读取 PDF 文件"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"读取 PDF 失败 {file_path}: {e}")
    return text


def read_docx(file_path):
    """读取 DOCX 文件"""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"读取 DOCX 失败 {file_path}: {e}")
        return ""


def read_excel(file_path):
    """
    读取 Excel 文件 (.xlsx, .xls)
    策略：读取所有 Sheet，将表格内容转换为字符串拼接
    """
    try:
        # 读取 Excel 的所有 sheet
        # dtype=str 强制将所有内容读取为字符串，防止数字/日期格式丢失或报错
        excel_data = pd.read_excel(file_path, sheet_name=None, dtype=str)

        full_text = []

        # 遍历每个 Sheet
        for sheet_name, df in excel_data.items():
            full_text.append(f"--- 表格名称: {sheet_name} ---")
            # 将 DataFrame 转为字符串，na_rep 用于填充空值
            sheet_text = df.to_string(index=False, na_rep="")
            full_text.append(sheet_text)

        return "\n".join(full_text)

    except Exception as e:
        print(f"读取 Excel 失败 {file_path}: {e}")
        return ""


def load_and_chunk_data(data_dir, chunk_size=300, chunk_overlap=50):
    """
    通用文档加载与切片函数
    支持: .txt, .pdf, .docx, .xlsx, .xls
    """
    documents = []

    # 遍历目录
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        # 跳过目录
        if os.path.isdir(file_path):
            continue

        content = ""

        # 根据后缀名选择读取方式
        if file.endswith(".txt"):
            content = read_txt(file_path)
        elif file.endswith(".pdf"):
            content = read_pdf(file_path)
        elif file.endswith(".docx"):
            content = read_docx(file_path)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            content = read_excel(file_path)
        else:
            continue  # 跳过不支持的文件格式

        # 简单清洗
        if content:
            content = content.strip()
            documents.append(content)

    # --- 切片逻辑 ---
    chunks = []
    for doc in documents:
        if len(doc) <= chunk_size:
            chunks.append(doc)
        else:
            start = 0
            while start < len(doc):
                end = start + chunk_size
                chunk = doc[start:end]
                chunks.append(chunk)

                start += chunk_size - chunk_overlap

                if start >= len(doc):
                    break

    print(f"成功加载并切片: {len(chunks)} 个文档块")
    return chunks