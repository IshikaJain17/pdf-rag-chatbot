<img width="2554" height="1400" alt="image" src="https://github.com/user-attachments/assets/a40b136b-5cbb-4a86-9fd3-68e70d0482ce" /># PDF RAG Chatbot Assistant

A powerful RAG (Retrieval-Augmented Generation) based chatbot that allows you to upload PDF documents and ask questions about them. The system automatically processes PDFs, stores them in a vector database, and uses OpenAI's models to provide intelligent answers based on your documents.

![PDF Chatbot](https://img.shields.io/badge/Python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-purple)


## Features

- 📄 **PDF Upload**: Drag & drop or browse to upload PDF documents
- 🔍 **Smart Search**: Automatically chunks documents and creates embeddings
- 💬 **Intelligent Q&A**: Ask questions and get accurate answers from your documents
- 📚 **Multi-Document Support**: Upload multiple PDFs and query across all of them
- 🗄️ **Persistent Storage**: Documents are stored in ChromaDB vector database
- 🎨 **Beautiful UI**: Modern, responsive web interface
- ⚡ **Real-time**: Fast responses with streaming support
- 🔒 **Secure**: All credentials stored in environment variables

## Project Structure

```
pdf-question-answer-chatbot-assistant/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration and environment variables
│   ├── main.py             # FastAPI application
│   ├── pdf_processor.py    # PDF text extraction and chunking
│   ├── vector_store.py     # ChromaDB vector database operations
│   └── rag_chain.py        # RAG question answering logic
├── static/
│   └── index.html          # Frontend web interface
├── uploads/                # Uploaded PDF files (auto-created)
├── chroma_db/              # Vector database storage (auto-created)
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Example environment configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- pip (Python package manager)

## Installation

### 1. Clone or Download the Project

```bash
cd "pdf question answer chatbot assistant"
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Edit the `.env` file and add your API keys:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# Vector Database Configuration (Pinecone)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=pdf-chatbot

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
MAX_TOKENS=1000
TEMPERATURE=0.7

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

## Running the Application

### Start the Server

```bash
# From the project root directory
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or alternatively:

```bash
python -m app.main
```

### Access the Application

Open your browser and go to:
```
http://localhost:8000
```

## Usage

1. **Upload a PDF**: 
   - Click on the upload area or drag & drop a PDF file
   - Wait for processing to complete

2. **Ask Questions**:
   - Type your question in the chat input
   - Press Enter or click the send button
   - Get answers based on your uploaded documents

3. **Manage Documents**:
   - View uploaded documents in the sidebar
   - Delete individual documents by clicking the X button
   - Clear all documents using the "Clear All" button

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the web interface |
| `/api/upload` | POST | Upload a PDF file |
| `/api/chat` | POST | Send a question and get an answer |
| `/api/chat/stream` | POST | Get streaming response |
| `/api/stats` | GET | Get collection statistics |
| `/api/documents/{filename}` | DELETE | Delete a specific document |
| `/api/documents` | DELETE | Clear all documents |
| `/api/health` | GET | Health check endpoint |

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `PINECONE_API_KEY` | Your Pinecone API key | Required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `pdf-chatbot` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `CHAT_MODEL` | OpenAI chat model | `gpt-4o-mini` |
| `MAX_TOKENS` | Max response tokens | `1000` |
| `TEMPERATURE` | Response creativity (0-1) | `0.7` |

## Supported Models

### Embedding Models
- `text-embedding-3-small` (recommended, cost-effective)
- `text-embedding-3-large` (higher quality)
- `text-embedding-ada-002` (legacy)

### Chat Models
- `gpt-4o-mini` (recommended, fast and affordable)
- `gpt-4o` (more capable, higher cost)
- `gpt-4-turbo` (powerful, higher cost)
- `gpt-3.5-turbo` (budget option)

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not configured"**
   - Make sure you've added your OpenAI API key to the `.env` file

2. **"No text extracted from PDF"**
   - The PDF might be scanned/image-based
   - Try a text-based PDF

3. **"Port already in use"**
   - Change the port in `.env` or use: `--port 8001`

4. **Module not found errors**
   - Make sure you've activated the virtual environment
   - Run `pip install -r requirements.txt` again

## Using Alternative Vector Databases

### ChromaDB (Local storage)

To use ChromaDB instead of Pinecone for local development:

1. Install ChromaDB: `pip install chromadb`
2. Update `.env`:
   ```env
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   CHROMA_COLLECTION_NAME=pdf_documents
   ```
3. Modify `vector_store.py` to use ChromaDB client (see commit history for the original ChromaDB implementation)

## Getting a Pinecone API Key

1. Go to [Pinecone.io](https://www.pinecone.io/) and sign up for a free account
2. Create a new project
3. Go to API Keys and copy your API key
4. The free tier includes 1 index with 100K vectors

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
