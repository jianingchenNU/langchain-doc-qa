# Document Q&A RAG System

A universal document question-answering system built with RAG (Retrieval-Augmented Generation) using LangChain and ChromaDB.

## Features

- Support for any text document Q&A
- Multi-document loading capability
- Interactive Gradio interface
- Vector similarity search
- Customizable configuration

## Installation

### Requirements

- Python 3.8+
- OpenAI API Key

### Setup Steps

1. Clone the repository
```bash
git clone https://github.com/yourusername/doc-qa-rag-system.git
cd doc-qa-rag-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure API Key
```bash
echo "your-openai-api-key" > apikey.txt
```

## Usage

### Basic Usage
```python
from doc_qa_system import DocumentQASystem

# Initialize system
qa_system = DocumentQASystem(api_key="your-api-key")

# Load document
qa_system.load_document("your_document.txt")

# Launch UI
qa_system.launch_ui()
```

### Load Multiple Documents
```python
qa_system.load_multiple_documents([
    "document1.txt",
    "document2.txt",
    "document3.txt"
])

qa_system.launch_ui()
```

### Programmatic Usage (No UI)
```python
answer, sources = qa_system.retrieve_and_answer("Your question?")
print(answer)
```

## Configuration
```python
qa_system = DocumentQASystem(
    api_key="your-api-key",
    model_name="gpt-4o-mini",      # Model to use
    chunk_size=1000,                # Text chunk size
    chunk_overlap=200,              # Chunk overlap
    n_results=3,                    # Number of chunks to retrieve
    collection_name="my_docs"       # Vector store collection name
)
```

## Supported File Formats

- .txt (Plain text)
- .md (Markdown)
- Any UTF-8 encoded text file

## Project Structure
```
doc-qa-rag-system/
├── doc_qa_system.py        # Main system class
├── main.py                 # Example usage
├── requirements.txt        # Dependencies
├── apikey.txt             # API Key (create yourself)
└── README.md              # Documentation
```

## Dependencies
```
langchain-core
langchain-openai
langchain-community
langchain-text-splitters
chromadb
gradio
openai
```

## How It Works

1. Load documents and split into chunks
2. Convert chunks to vectors and store in ChromaDB
3. Retrieve most relevant chunks when user asks a question
4. Send retrieved context and question to LLM to generate answer

## FAQ

**Q: How to change the model?**
A: Set the `model_name` parameter during initialization, e.g., `model_name="gpt-4"`

**Q: How to adjust retrieval accuracy?**
A: Modify `n_results` to retrieve more chunks, or adjust `chunk_size` to change chunk granularity

**Q: Does it support non-English documents?**
A: Yes, ensure the file uses UTF-8 encoding

## License

MIT License
