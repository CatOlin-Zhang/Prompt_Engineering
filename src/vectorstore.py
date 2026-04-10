# -*- coding: utf-8 -*-
"""
向量数据库构建与混合检索 (无 LangChain 版)
使用: Scikit-learn (Lexical) + Sentence-Transformers (Embedding)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, chunks, embedding_model_name="all-MiniLM-L6-v2"):
        """
        初始化检索器
        :param chunks: 文档切片列表 (List of strings)
        """
        self.chunks = chunks
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        print("正在构建索引...")
        # 1. 准备 Lexical 检索 (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([chunk for chunk in chunks])
        
        # 2. 准备 Embedding 检索 (Dense Vector)
        # 一次性将所有文档块转化为向量，存入内存
        self.chunk_embeddings = self.embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        
        print("索引构建完成！")

    def get_context(self, query, top_k=3, hybrid_alpha=0.5):
        """
        执行混合检索
        :param query: 用户问题
        :param top_k: 返回多少个片段
        :param hybrid_alpha: 混合权重 (0.5 = 50% 关键词 + 50% 向量)
        :return: 拼接好的上下文字符串
        """
        
        # --- 1. Lexical Retrieval (TF-IDF) ---
        query_tfidf = self.tfidf_vectorizer.transform([query])
        # 计算余弦相似度 (1xN)
        lexical_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # --- 2. Embedding Retrieval ---
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        # 计算余弦相似度 (1xN)
        semantic_scores = cosine_similarity(query_embedding, self.chunk_embeddings).flatten()
        
        # --- 3. Hybrid Search (加权融合) ---
        # 归一化分数 (简单处理：防止数值差异过大)
        # 实际项目中可以用 MinMaxScaler，这里为了演示简单直接用权重相加
        final_scores = (hybrid_alpha * lexical_scores) + ((1 - hybrid_alpha) * semantic_scores)
        
        # 获取分数最高的索引
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # --- 4. 格式化输出 ---
        context_parts = []
        for i, idx in enumerate(top_indices):
            # 获取对应的文本块
            text = self.chunks[idx]
            # 获取分数用于调试
            score = final_scores[idx]
            context_parts.append(f"[{i+1}] (Score: {score:.2f}) {text}")
            context_text = "\n\n".join(context_parts)
        return context_text, [self.chunks[idx] for idx in top_indices]