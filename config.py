# -*- coding: utf-8 -*-
"""
项目全局配置文件
"""
import os

# 模型配置 (作业要求: <=12B parameters)
LOCAL_MODEL = "gemma3:4b" # 或 gemma3:4b
OLLAMA_HOST = "http://localhost:11434" # 默认端口

# 路径配置
DATA_DIR = "./data/raw"
CHROMA_DB_DIR = "./chroma_db" # 向量数据库保存路径

# RAG 参数
CHUNK_SIZE = 300 # 切片大小 (Token Efficiency 关键点)
CHUNK_OVERLAP = 50 # 切片重叠
TOP_K_CONTEXT = 3 # 检索返回的上下文数量