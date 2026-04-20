"""
配置文件，包含所有项目参数
"""

# 模型配置
LOCAL_MODEL = "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"

# 文件路径
DATA_DIR = "./data"
CHROMA_DB_DIR = "./chroma_db"

# RAG配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_CONTEXT = 4

# 模型参数
TEMPERATURE = 0.3
NUM_CTX = 4096

# 评分权重
HYBRID_ALPHA = 0.5

# 内存配置
MAX_HISTORY = 3

# 评判模型
JUDGE_MODEL = "gemma3:4b-cloud"