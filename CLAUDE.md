# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Japanese RAG (Retrieval-Augmented Generation) Question-Answering system that implements semantic coverage analysis for evaluating Q&A datasets against documents. The system uses OpenAI embeddings and Qdrant vector database for similarity search and coverage metrics calculation.

## Development Commands

### Environment Setup
```bash
# Initial setup (installs packages and configures environment)
python setup.py

# Install dependencies
pip install -r requirements.txt

# Start Qdrant vector database
docker-compose -f docker-compose/docker-compose.yml up -d

# Register data to Qdrant
python a30_qdrant_registration.py --recreate --limit 100
```

### Running the Application
```bash
# Start the Qdrant server management script
python server.py

# Run Streamlit search UI
streamlit run a50_rag_search_local_qdrant.py

# Run example semantic coverage analysis
python example.py
```

### Code Quality
```bash
# Run ruff linter (no configuration file exists yet)
ruff check .

# Format code with ruff
ruff format .
```

## Architecture

### Core Components

1. **SemanticCoverage** (`rag_qa.py`): Main class implementing document chunking and semantic coverage calculation
   - Creates semantic chunks from documents
   - Generates embeddings for documents and Q&A pairs
   - Calculates coverage metrics using cosine similarity
   - Supports Japanese text processing with sentence boundary detection

2. **Helper Modules**:
   - `helper_api.py`: OpenAI API integration, model configuration, and cost tracking
   - `helper_rag.py`: RAG data preprocessing, configuration management (AppConfig class)
   - `helper_st.py`: Streamlit utilities for customer support FAQ processing

3. **Data Management Scripts** (a-prefixed files):
   - `a01_load_set_rag_data.py`: Load and set RAG data
   - `a02_set_vector_store_vsid.py`: Configure vector store IDs
   - `a03_rag_search_cloud_vs.py`: Search cloud vector stores
   - `a30_qdrant_registration.py`: Register data to Qdrant
   - `a35_qdrant_truncate.py`: Truncate Qdrant collections
   - `a40_show_qdrant_data.py`: Display Qdrant data
   - `a50_rag_search_local_qdrant.py`: Streamlit UI for local Qdrant search

4. **Infrastructure**:
   - `server.py`: Qdrant server health checks and startup management
   - `docker-compose/docker-compose.yml`: Containerized Qdrant deployment

### Data Flow

1. Documents are split into semantic chunks preserving sentence boundaries
2. OpenAI embeddings are generated for chunks and Q&A pairs
3. Embeddings are stored in Qdrant vector database
4. Coverage analysis compares Q&A embeddings against document chunks
5. Results are presented via Streamlit UI or API endpoints

### Model Configuration

The system supports extensive OpenAI models (configured in `config.yml`):
- GPT-4o series (including mini and audio variants)
- GPT-4.1, GPT-5 series
- O-series models (o1, o3, o4 with mini variants)
- Embedding models (text-embedding-3-small/large)

## Environment Variables

Required in `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333  # Optional, defaults to localhost
PG_CONN_STR=postgresql://...       # Optional, for PostgreSQL integration
```

## Key Implementation Details

- **Japanese Text Processing**: Uses regex patterns for Japanese sentence splitting
- **Chunking Strategy**: Semantic chunking with 200 token limit per chunk
- **Embedding Model**: Default is "text-embedding-3-small"
- **Coverage Threshold**: 0.8 cosine similarity for matching Q&A to chunks
- **Token Counting**: Uses tiktoken with "cl100k_base" encoding

## Dependencies

Main packages:
- `openai>=1.100.2`: API client for embeddings and chat
- `qdrant-client>=1.15.1`: Vector database client
- `streamlit>=1.48.1`: Web UI framework
- `fastapi>=0.115.6`: API server framework
- `tiktoken`: Token counting for chunk size management
- `scikit-learn`: Cosine similarity calculations

## Important Notes

- No formal test suite exists - consider adding pytest when implementing new features
- The codebase uses Japanese variable names and comments in some places
- Old implementations are archived in the `old_code/` directory
- Qdrant must be running before using any data registration or search functionality