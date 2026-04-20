"""
生成器模块
负责发送请求、构建复杂提示(包含CoT)和管理生成参数
"""
import tiktoken
import ollama


class OllamaGenerator:
    def __init__(self, local_model, temperature, num_ctx):
        self.model = local_model
        # 保存默认配置
        self.default_options = {"temperature": temperature, "num_ctx": num_ctx}

        # 初始化Token编码器
        try:
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.token_encoder = tiktoken.get_encoding("p50k_base")

    def generate_response(self, final_prompt, temperature=None):
        """
        从模型生成响应
        :param final_prompt: 发送给模型的提示字符串
        :param temperature: 可选，覆盖默认温度(例如，在评判模式下设为0)
        """
        # 准备配置：如果提供了温度则覆盖，否则使用默认值
        current_options = self.default_options.copy()
        if temperature is not None:
            current_options["temperature"] = temperature

        try:
            # 调用Ollama API
            response = ollama.generate(
                model=self.model,
                prompt=final_prompt,
                options=current_options
            )
            return response['response']
        except Exception as e:
            return f" Ollama Connection Error: {e}"

    def build_rag_prompt(self, query, context, chat_history=""):
        """
        提示组装：组装系统指令、历史、当前查询和CoT推理过程
        """
        prompt = f"""
        # Role: HKBU Study Companion Expert
        # Goal: Answer the question based on the provided Context. If no relevant information is found in the Context, answer "Cannot answer based on available materials".
        
        # Instructions:
        # 1. Follow Chain-of-Thought (CoT) reasoning process
        # 2. Explicitly state your reasoning steps
        # 3. Cite sources from context when applicable
        # 4. Provide precise, concise answers
        
        ## Example Format (Follow this structure exactly):
        # Context: [1] Course Policies: Lab assignments are due every Monday 13:30. [2] Late submission without prior approval will not be accepted.
        # Question: What happens if I submit my lab assignment late?
        # Reasoning: 1. Question asks about consequences of late submission. 2. Context [2] states late submissions without approval are not accepted. 3. This means late work will not be graded.
        # Answer: Late submissions without prior approval will not be accepted and will not be graded [2].
        
        ## Advanced Reasoning Example:
        # Context: [3] From Year 2, students are encouraged to meet advisors as needed. [4] Students must meet advisor under specific conditions: (i) major change; (ii) advisor change; (iii) academic probation; (iv) opt out of TSM/SM.
        # Question: As a Year 2 student, do I need to meet my advisor?
        # Reasoning: 1. Identify situation: Year 2 student. 2. General rule: [3] encourages meetings but not mandatory. 3. Specific requirements: [4] mandates meetings under certain conditions. 4. Synthesize: Differentiate between encouraged vs. required scenarios.
        # Answer: As a Year 2 student, meeting with your advisor is encouraged but not mandatory unless you're changing your major, experiencing an advisor change, on academic probation, or opting out of TSM/SM [3,4].
        
        ## Context (Reference)
        {context}
        
        ## Conversation History (if any)
        {chat_history}
        
        ## Final Task
        Apply the reasoning pattern above to answer the question.
        Question: {query}
        Reasoning: [Provide step-by-step logical reasoning]
        Answer: [Provide final answer with source citations if applicable]
        """
        return prompt

    def build_tool_augmented_prompt(self, query, context, chat_history="", tools_info=""):
        """
        构建工具增强型提示（ReAct风格）
        """
        prompt = f"""
        # Role: HKBU Study Companion with Tool Access
        # Goal: Answer questions using provided context AND available tools when necessary
        
        ## Available Tools:
        {tools_info}
        
        ## Reasoning Framework (ReAct Style):
        # 1. Think: Analyze what information is needed
        # 2. Act: Decide whether to use tools or rely on provided context
        # 3. Observe: Process the information obtained
        # 4. Think: Synthesize the information
        # 5. Respond: Provide final answer
        
        ## Example:
        # Context: [1] Office hours: Mon-Wed 2-4pm. [2] Email: cs@hkbu.edu.hk
        # Query: When are CS office hours?
        # Think: The context contains office hours information.
        # Observe: Context [1] has the required information.
        # Answer: CS office hours are Monday and Wednesday from 2-4pm [1].
        
        ## Context (Reference)
        {context}
        
        ## Conversation History
        {chat_history}
        
        ## Task
        Question: {query}
        Think: [Step-by-step analysis]
        Act: [Tool usage if needed, otherwise proceed with context]
        Observe: [Information processing]
        Answer: [Final response with proper citation]
        """
        return prompt

    def count_tokens(self, text):
        """
        计算文本的Token数量
        """
        if not text:
            return 0
        return len(self.token_encoder.encode(text))