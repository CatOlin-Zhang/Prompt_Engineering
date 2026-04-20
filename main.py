"""
主程序
协调各个模块执行增强版RAG系统的完整流程
"""
import time
from config import *
from data_loader import load_and_chunk_data
from retriever import RAGRetriever, display_retrieved_chunks
from memory import ConversationMemory
from generator import OllamaGenerator
from judge import judge
from advanced_demo import run_advanced_demo
import tiktoken


def count_tokens(text):
    """
    计算文本的Token数量
    """
    try:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    except:
        token_encoder = tiktoken.get_encoding("p50k_base")

    if not text:
        return 0
    return len(token_encoder.encode(text))


def main():
    """
    主函数，执行增强版RAG系统的完整流程
    """
    print("\033[1mStarting Enhanced RAG System\033[0m")
    print("Features: CoT Reasoning, Tool Augmentation, Advanced Evaluation")

    # 询问用户是否启用LLM评估
    enable_judge = input("Enable LLM-based comparative evaluation? (y/n, default y): ").strip().lower()
    if enable_judge in ['', 'y', 'yes']:
        enable_judge = True
        print(" LLM evaluation enabled")
    else:
        enable_judge = False
        print(" LLM evaluation disabled")

    # 加载和分块数据
    print("Loading and chunking data...")
    chunks = load_and_chunk_data(DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    # 调试信息：检查chunks的结构
    print(f"Debug: First chunk type = {type(chunks[0]) if chunks else 'empty'}")
    if chunks:
        print(f"Debug: First chunk = {str(chunks[0])[:100]}...")

    # 初始化检索器
    print("Initializing retriever...")
    retriever = RAGRetriever(chunks)

    # 初始化生成器
    print("Initializing generator...")
    generator = OllamaGenerator(LOCAL_MODEL, TEMPERATURE, NUM_CTX)

    # 初始化内存
    print("Initializing memory...")
    memory = ConversationMemory()

    print(f"System Ready | Model: {LOCAL_MODEL}")
    print("="*130)

    # 主交互循环
    while True:
        try:
            user_input = input("\nUser (Type 'exit' to quit): ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            print(f"\n🔍 Retrieving for question: '{user_input}'...")

            # --- 获取历史记忆 ---
            chat_history = memory.get_formatted_history()

            results = {}

            # --- 定义三种检索策略 ---
            strategies = [
                ("Lexical Retrieval (TF-IDF)", lambda q: retriever.retrieve_lexical(q, top_k=TOP_K_CONTEXT)),
                ("Neural Retrieval (Embedding)", lambda q: retriever.retrieve_embedding(q, top_k=TOP_K_CONTEXT)),
                ("Hybrid Retrieval (Optimal)", lambda q: retriever.retrieve_hybrid(q, top_k=TOP_K_CONTEXT, alpha=HYBRID_ALPHA)),
            ]

            # 执行检索、显示块、生成响应
            for strat_name, retrieve_fn in strategies:
                print(f"\n--- [{strat_name}] Retrieval Details ---")

                # 开始计时
                t0 = time.perf_counter()
                retrieved_items = retrieve_fn(user_input)

                if not retrieved_items:
                    elapsed = time.perf_counter() - t0
                    print("No relevant chunks found.")
                    results[strat_name] = {
                        "response": "No relevant content",
                        "context": "",
                        "tokens": 0,
                        "latency": elapsed,
                        "strategy": strat_name
                    }
                    print(f"    Total Time: {elapsed:.2f}s")
                    continue

                # 1. 显示块排名和分数（使用改进的显示函数）
                context_parts = []
                display_retrieved_chunks(retrieved_items, max_preview_length=800)  # 显示更长的预览

                for text, _ in retrieved_items:
                    context_parts.append(str(text))  # 确保是字符串

                # 2. 构建上下文并生成
                full_context = "\n\n".join(context_parts)

                # 使用增强版提示
                prompt = generator.build_rag_prompt(user_input, full_context, chat_history)

                response = generator.generate_response(prompt)

                # 结束计时
                elapsed = time.perf_counter() - t0

                # 3. 计算Token
                total_tokens = count_tokens(prompt) + count_tokens(response)

                # 4. 保存结果
                results[strat_name] = {
                    "response": response,
                    "context": full_context,
                    "tokens": total_tokens,
                    "latency": elapsed,
                    "strategy": strat_name
                }

                print(f"\n--\033[1mGenerated Response\033[0m--")
                print(response)  # 显示完整回复，不再截断
                print(f"... (Consumed {total_tokens} tokens) ...\n")
                print(f"⏱  Total Time: {elapsed:.2f}s")
                print("-" * 130)

            # 最终统计
            print("\n \033[1mPerformance Comparison:\033[0m")
            print(f"{'Strategy':<25} {'Tokens':<10} {'Latency(s)':<12} {'Efficiency':<12}")
            print("-" * 65)

            for name, data in results.items():
                tokens = data.get('tokens', 0)
                latency = data.get('latency', 0.0)
                efficiency = tokens / latency if latency > 0 else 0
                print(f"{name:<25} {tokens:<10} {latency:<12.2f} {efficiency:<12.2f}")

            print("-" * 65)

            # 更新内存
            final_response = results.get("Hybrid Retrieval (Optimal)", {}).get("response", "No answer")

            # 添加当前问答对到历史
            memory.add_message("User", user_input)
            memory.add_message("Assistant", final_response)

            print(f"💬 Conversation saved to memory (Current length: {len(memory.history)})")

            # LLM评判评估（可选）
            if enable_judge:
                print("\n" + "=" * 130)
                print("  \033[1mStarting LLM Judge for comparative evaluation...\033[0m")
                print("-" * 130)

                r_lex = results.get("Lexical Retrieval (TF-IDF)", {}).get("response", "No content")
                r_emb = results.get("Neural Retrieval (Embedding)", {}).get("response", "No content")
                r_hyb = results.get("Hybrid Retrieval (Optimal)", {}).get("response", "No content")

                try:
                    verdict = judge(user_input, r_lex, r_emb, r_hyb, activate=True)
                    print(f"Evaluation Result: {verdict}")
                except Exception as e:
                    print(f"Evaluation failed due to error: {e}")
                    print("Continuing without evaluation...")
            else:
                print("\n  Skipping LLM-based evaluation as requested")

        except KeyboardInterrupt:
            print("\n Program exited by user.")
            break
        except Exception as e:
            print(f"\n Error occurred: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Welcome to HKBU Study Companion RAG System!")
    print("Choose mode:")
    print("1. Standard RAG (default)")
    print("2. Advanced Demo")

    choice = input("Enter choice (1 or 2, default 1): ").strip()

    if choice == "2":
        run_advanced_demo()

    # 总是运行主系统
    main()