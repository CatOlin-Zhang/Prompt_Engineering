"""
COMP4146 Project - Main Tester (Integrated Data Generator)
修复了数据加载为 0 的问题，确保数据在运行前一定存在。
"""

import os
import time
from pathlib import Path


# --- 0. 强制数据生成模块 (确保数据一定存在) ---
def ensure_data_exists():
    data_dir = Path("./data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 定义必须的文件内容
    files_content = {
        "Course_Syllabus.txt": """
COMP4146 Prompt Engineering for Generative AI
Grading Breakdown:
Lab assignments: 20 percent
In-class quiz: 10 percent
Group project: 20 percent
Final project: 50 percent
The final project includes a group presentation and a personal report.
""",
        "Course_Policies.txt": """
Course Policies:
Lab assignments are due every Monday 13:30.
Late submission without prior approval will not be accepted.
Attendance is recommended for all lab sessions.
""",
        "Book_Reviews.txt": """
User review snippets about The Beach:
I enjoy travel stories with moral ambiguity and flawed characters.
I do not like slow starts, but I like high psychological tension.
I prefer vivid setting descriptions over action-heavy writing.
"""
    }

    # 写入文件
    for filename, content in files_content.items():
        file_path = data_dir / filename
        file_path.write_text(content.strip(), encoding="utf-8")

    print(f"✅ 数据检查完成：确保 {len(files_content)} 个文件已存在于 {data_dir.absolute()}")


# 先运行数据生成
ensure_data_exists()
print("-" * 60)

# --- 1. 导入模块 ---
# (注意：如果这里报错，说明环境配置还有问题)
try:
    from src.data_loader import load_and_chunk_data
    from src.vectorstore import RAGRetriever
    from src.generator import OllamaGenerator
    from src.conversation import ConversationMemory
    from src.evaluator import estimate_tokens
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请检查 src 文件夹是否存在，以及 __init__.py 是否为空。")
    exit()

# --- 2. 全局配置 ---
DATA_DIR = "./data/raw"
TEST_QUERIES = [
    "What is the grading breakdown for the final project?",
    "When are lab assignments due?",
    "I like psychological tension but hate slow starts. Should I read The Beach?"
]


def main():
    print("🚀 开始初始化 HKBU Study Companion 测试...\n")
    start_time = time.time()

    # --- 阶段 1: 数据加载与切片 ---
    print("[1/5] 正在加载和切分文档...")
    try:
        # 再次确认目录路径
        if not os.path.exists(DATA_DIR):
            print(f"❌ 错误: 目录 {DATA_DIR} 不存在！")
            return

        chunks = load_and_chunk_data(DATA_DIR, chunk_size=300, chunk_overlap=50)

        if len(chunks) == 0:
            print(f"❌ 错误: 目录 {DATA_DIR} 是空的，或者文件读取失败。")
            print(f"   请检查: {[f for f in os.listdir(DATA_DIR)]}")
            return

        print(f"   成功加载 {len(chunks)} 个文本块。\n")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        return

    # --- 阶段 2: 初始化 RAG 检索器 ---
    print("[2/5] 正在构建向量数据库 (RAG)...")
    try:
        retriever = RAGRetriever(chunks=chunks)
        print("   向量数据库构建完成。\n")
    except Exception as e:
        print(f"   RAG 初始化失败: {e}")
        return

    # --- 阶段 3: 初始化生成器 ---
    print("[3/5] 正在连接本地 Ollama 模型...")
    try:
        generator = OllamaGenerator()
        print(f"   成功连接模型: {generator.model}\n")
    except Exception as e:
        print(f"   模型连接失败: {e}")
        print("   请确保 Ollama 正在运行。")
        return

    # --- 阶段 4: 初始化对话记忆 ---
    memory = ConversationMemory(max_history=3)
    print("[4/5] 对话记忆系统初始化完成。\n")

    # --- 阶段 5: 运行测试用例 ---
    print("[5/5] 开始运行测试用例...\n")
    print("-" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n💬 测试 {i}: {query}")

        input_token_est = estimate_tokens(query)
        print(f"   📊 预估输入 Token: {input_token_est}")

        try:
            # 检索
            context, _ = retriever.get_context(query)
            print(f"   🔍 检索成功 (上下文长度: {len(context)} 字符)")

            # 生成
            final_prompt = generator.build_rag_prompt(query, context, memory.get_formatted_history())
            print("   🤖 助手正在思考...", end="\r")
            response = generator.generate_response(final_prompt)
            print(f"   ✅ 助手: {response[:100]}...")  # 只打印前100个字符避免刷屏

            # 记忆
            memory.add_message("User", query)
            memory.add_message("Assistant", response)

        except Exception as e:
            print(f"   ❌ 生成失败: {e}")

        print("-" * 60)

    total_time = time.time() - start_time
    print(f"\n🎉 测试完成！总耗时: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()