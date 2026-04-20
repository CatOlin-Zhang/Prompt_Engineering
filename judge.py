"""
评判模块
使用指定模型比较不同检索策略的响应质量
"""
import ollama
from config import JUDGE_MODEL, TEMPERATURE
import requests


def judge(question, response_lexical, response_neural, response_hybrid, activate=True):
    """
    使用LLM评判三种响应的质量
    :param activate: 控制是否激活LLM评估，默认True
    """
    if not activate:
        return "LLM Evaluation: Skipped as requested"

    judge_prompt = f"""
    # Task: Compare and rank the quality of three responses to the same question
    # Question: {question}
    
    ## Response A (Lexical/TF-IDF retrieval): 
    {response_lexical}
    
    ## Response B (Neural/Embedding retrieval):
    {response_neural}
    
    ## Response C (Hybrid retrieval):
    {response_hybrid}
    
    # Evaluation Criteria:
    # 1. Accuracy: How factually correct is the response?
    # 2. Relevance: How well does it address the specific question?
    # 3. Completeness: Does it provide sufficient information?
    # 4. Clarity: Is the response clear and well-structured?
    
    # Instruction: Rank the responses from best to worst (e.g., "Ranking: C > A > B")
    # Provide brief justification for your ranking
    # Format your response as: "Ranking: [best] > [middle] > [worst]\nJustification: [your explanation]"
    """

    try:
        # 设置更长的超时时间
        response = ollama.generate(
            model=JUDGE_MODEL,
            prompt=judge_prompt,
            options={"temperature": TEMPERATURE},
            keep_alive=-1  # 保持连接
        )
        return response['response']
    except requests.exceptions.Timeout:
        return " Judge Error: Request timed out after waiting for response"
    except requests.exceptions.ConnectionError:
        return " Judge Error: Connection error during evaluation"
    except Exception as e:
        return f" Judge Error: {e}"