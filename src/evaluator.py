"""评估 (Token效率与对比实验)"""
# -*- coding: utf-8 -*-
"""
评估模块
"""
#TODO 对代表性查询进行基础评估，包括至少一项质量比较和一项词元使用比较。（主题4）
def estimate_tokens(text):
    """粗略估算 Token 数量 (用于 Token Efficiency 分析)"""
    # 简单估算: 1个 Token 约等于 4个英文字符或 1.3个中文字符
    # 这里用简单方法: 字符数 / 4
    return len(text) // 4

def compare_rag_vs_no_rag(generator, query, context):
    """
    对比实验: RAG vs No-RAG
    """
    # Test 1: No-RAG (直接问模型)
    no_rag_prompt = f"Question: {query}\nAnswer:"
    no_rag_time = ... # 这里需要你计时
    no_rag_response = generator.generate_response(no_rag_prompt)
    no_rag_tokens = estimate_tokens(no_rag_prompt + no_rag_response)

    # Test 2: RAG (带上下文)
    rag_prompt = generator.build_rag_prompt(query, context)
    rag_time = ...
    rag_response = generator.generate_response(rag_prompt)
    rag_tokens = estimate_tokens(rag_prompt + rag_response)

    print(f"No-RAG Tokens: {no_rag_tokens}, RAG Tokens: {rag_tokens}")
    # 这里可以人工或自动对比回答质量
    return {"no_rag": no_rag_tokens, "rag": rag_tokens}