# -*- coding: utf-8 -*-
"""
原生 Python 文档加载与切片
"""
import os

def load_and_chunk_data(data_dir, chunk_size=300, chunk_overlap=50):
    """
    1. 读取 .txt 文件 (因为我们在生成数据时用了 txt)
    2. 简单的按字符切片
    """
    documents = []

    # 遍历目录下的所有 txt 文件
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单清洗：去除多余空行
                content = content.strip()
                documents.append(content)

    # 简单的切片逻辑 (按字符长度)
    # 注意：这里没有做复杂的语义分割，适合演示
    chunks = []
    for doc in documents:
        # 如果文档太短，直接加入
        if len(doc) <= chunk_size:
            chunks.append(doc)
        else:
            # 滑动窗口切片
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