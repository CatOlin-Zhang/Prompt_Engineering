"""Ollama生成与Prompt组装"""
# -*- coding: utf-8 -*-
"""
Ollama 生成模块
"""
import ollama
from config import LOCAL_MODEL, OLLAMA_HOST

class OllamaGenerator:
    def __init__(self, model_name=LOCAL_MODEL):
        self.model = model_name
        # TODO 生成控制需包含关键参数（如温度参数和输出长度）并在报告中简要说明设置依据。
        self.options = {
            "temperature": 0.3, # 低温度保证回答稳定，减少幻觉
            "num_ctx": 2048,    # 上下文长度
        }


    def generate_response(self, final_prompt):
        """
        使用 Raw Generation API
        """
        response = ollama.generate(
            model=self.model,
            prompt=final_prompt,
            options=self.options
        )
        return response['response']

    def build_rag_prompt(self, query, context, chat_history=""):
        """
        Prompt Assembly: 组装系统指令、历史记录和当前问题
        """
        #TODO 提示组装需包含清晰指令、上下文信息及应答格式控制（例如角色/任务框架设定、明确约束条件以及向最终答案的清晰过渡）
        prompt = f"""
        # Role: HKBU Study Companion
        # Goal: 根据提供的 Context 回答问题，如果 Context 中没有相关信息，请回答“根据现有资料无法回答”。

        ## Context (Reference)
        {context}

        ## Conversation History (if any)
        {chat_history}

        ## Final Task
        请用中文回答以下问题，并在回答中引用对应的 Context 编号（例如 [1]）。
        Question: {query}
        Answer:
        """
        return prompt