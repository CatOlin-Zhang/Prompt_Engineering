"""
检索器模块
实现TF-IDF、嵌入向量和混合检索
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from config import TOP_K_CONTEXT, HYBRID_ALPHA
import re


class RAGRetriever:
    """
    RAG检索器类
    实现多种检索策略
    """
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # 为TF-IDF构建矩阵
        print("Processing data format...")

        # 检查chunks的数据结构并适当地提取文本
        if isinstance(chunks, list) and len(chunks) > 0:
            if isinstance(chunks[0], dict):
                # 如果chunks是字典列表，如{"text": "..."}
                texts = [chunk.get('text', '') if isinstance(chunk, dict) else str(chunk) for chunk in chunks]
            elif isinstance(chunks[0], str):
                # 如果chunks是字符串列表
                texts = [str(chunk) for chunk in chunks]
            else:
                # 其他情况转换为字符串
                texts = [str(chunk) for chunk in chunks]
        else:
            texts = []

        # 过滤掉空文本
        texts = [text for text in texts if text.strip()]

        if not texts:
            raise ValueError("No valid text chunks found for indexing")

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # 为嵌入向量构建向量库
        print("Building Embedding vector store...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chunk_embeddings = self.embedding_model.encode(texts)
        print("Initialization complete.")

    def retrieve_lexical(self, query, top_k=TOP_K_CONTEXT):
        """
        词汇检索 (TF-IDF)
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [(self.chunks[i], similarities[i]) for i in top_indices if i < len(self.chunks)]
        return results

    def retrieve_embedding(self, query, top_k=TOP_K_CONTEXT):
        """
        嵌入向量检索
        """
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [(self.chunks[i], similarities[i]) for i in top_indices if i < len(self.chunks)]
        return results

    def retrieve_hybrid(self, query, top_k=TOP_K_CONTEXT, alpha=HYBRID_ALPHA):
        """
        混合检索 (词汇 + 嵌入向量)
        alpha: 词汇检索权重 (1-alpha): 嵌入向量权重
        """
        # 获取词汇检索分数
        lexical_results = self.retrieve_lexical(query, top_k=len(self.chunks))
        lexical_scores = {i: score for i, (_, score) in enumerate(lexical_results)}

        # 获取嵌入向量检索分数
        embedding_results = self.retrieve_embedding(query, top_k=len(self.chunks))
        embedding_scores = {i: score for i, (_, score) in enumerate(embedding_results)}

        # 归一化分数
        max_lexical_score = max(lexical_scores.values()) if lexical_scores else 1
        max_embedding_score = max(embedding_scores.values()) if embedding_scores else 1

        # 计算混合分数
        hybrid_scores = {}
        for i in range(len(self.chunks)):
            lex_score = lexical_scores.get(i, 0) / max_lexical_score if max_lexical_score > 0 else 0
            emb_score = embedding_scores.get(i, 0) / max_embedding_score if max_embedding_score > 0 else 0
            hybrid_scores[i] = alpha * lex_score + (1 - alpha) * emb_score

        # 排序并返回top_k
        sorted_indices = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:top_k]
        results = [(self.chunks[i], hybrid_scores[i]) for i in sorted_indices if i < len(self.chunks)]
        return results


def truncate_text_for_display(text, max_length=500):
    """
    截断文本用于显示，保留句子完整性
    """
    if len(text) <= max_length:
        return text

    # 尝试在句子边界处截断
    truncated = text[:max_length]

    # 查找最后一个句号、感叹号或问号
    sentence_endings = ['.', '!', '?', '\n']
    for ending in sentence_endings:
        last_pos = truncated.rfind(ending)
        if last_pos != -1:
            truncated = truncated[:last_pos+1]
            break

    # 如果没有找到句子边界，则按字符截断
    if len(truncated) == max_length:
        truncated = truncated + "..."
    else:
        truncated = truncated + "..."

    return truncated


def display_retrieved_chunks(retrieved_items, max_preview_length=800):
    """
    显示检索到的文档块，带有更详细的预览
    """
    for i, (text, score) in enumerate(retrieved_items):
        preview = truncate_text_for_display(str(text), max_preview_length)
        print(f"   #{i+1} [Score: {score:.4f}] {preview}")
        print()  # 添加空行以改善可读性)