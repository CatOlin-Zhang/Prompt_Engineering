"""对话记忆管理"""
# -*- coding: utf-8 -*-
"""
对话记忆管理
"""
#TODO 适当时实施对话管理（历史记录处理和/或系统消息控制）
class ConversationMemory:
    def __init__(self, max_history=3):
        # 限制历史轮次以保证 Token Efficiency
        self.max_history = max_history
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        # 限制长度，只保留最近的对话
        if len(self.history) > self.max_history * 2: # *2 因为包含 user 和 ai
            self.history = self.history[-(self.max_history * 2):]

    def get_formatted_history(self):
        # 格式化历史记录供 Prompt 使用
        formatted = ""
        for msg in self.history:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted

    def clear(self):
        self.history = []