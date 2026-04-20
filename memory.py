"""
对话记忆模块
管理对话历史记录，使AI能够记住之前的对话内容
"""


class ConversationMemory:
    def __init__(self, max_history=3):
        """
        初始化记忆模块
        :param max_history: 最多保留的对话轮数
        """
        self.max_history = max_history
        self.history = []

    def add_message(self, role, content):
        """
        添加新消息到历史记录，并管理滑动窗口
        自动移除超出限制的旧消息
        """
        self.history.append({"role": role, "content": content})
        # 保留最后max_history * 2条消息(用户+AI对)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_formatted_history(self):
        """
        返回格式化的对话历史
        格式: "角色: 内容"
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])