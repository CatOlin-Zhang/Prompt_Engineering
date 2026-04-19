<div align="center">

# Study Companion RAG System



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-green.svg)](https://ollama.ai/)


</div>

---

##  Features

- **Multi-Strategy Retrieval** : Combines lexical (TF-IDF) and neural (embedding) search with hybrid approach
- **Chain-of-Thought Reasoning** : Step-by-step logical processing for complex queries  
- **Conversational Memory** : Persistent context across multi-turn interactions
- **Performance Evaluation** : Built-in comparison of different retrieval strategies
- **Token Efficiency** : Optimized prompting and response generation
- **File Format Support** : PDF, DOCX, TXT, PPTX document processing
- **Local LLM Integration** : Uses Ollama for privacy-preserving inference

---

##  Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [Ollama](https://ollama.ai/) installed and running
- Required models (e.g., `llama3:8b-instruct-q8_0`)
- Internet connection for initial setup

---

## Install Python Dependencies
```bash
 pip install -r requirements.txt
 ```
Required packages:
- ollama
- scikit-learn
- sentence-transformers
- tiktoken
- PyPDF2
- python-docx
- openpyxl
- python-pptx
## Set Up Ollama
```bash
  # Install Ollama (visit https://ollama.ai for instructions)
  # Pull required models
ollama pull models_name
  # Two models are needed to enable all the functions of the agent. 
 ```
## Prepare Data
```bash
  mkdir -p ./data
  cp your_documents/* ./data/
  
 ```
Supported formats: .pdf, .docx, .txt

## Configuration

Edit `config.py` to customize system behavior:

```python
LOCAL_MODEL = "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_CONTEXT = 4

TEMPERATURE = 0.3
NUM_CTX = 4096

JUDGE_MODEL = "gemma3:4b-cloud"
```

