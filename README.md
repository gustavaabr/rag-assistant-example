# RAG Assistant

A minimal Retrieval-Augmented Generation (RAG) assistant: PDFs stored in S3 are chunked, embedded, and indexed in Pinecone, then queried through a Streamlit chat UI backed by an OpenAI LLM.

## Architecture

S3 (PDFs) → PyPDFLoader + text splitter → OpenAI Embeddings → Pinecone index
                                                                    ↓
                                    Streamlit chat UI ← LLM answer ← retriever

- `src/ingest.py` — downloads PDFs from an S3 bucket, splits them into chunks, embeds them, and upserts them into Pinecone.
- `src/llm_chain.py` — retrieves relevant chunks from Pinecone for a question and generates an answer with citations.
- `app.py` — Streamlit chat interface.

## Setup

1. **Clone and install dependencies**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

2. Configure environment variables

2. Copy .env_example to .env and fill in your real values:
cp .env_example .env

| Variable                                               | Description                                                          |
|--------------------------------------------------------|----------------------------------------------------------------------|
| OPENAI_API_KEY                                         | OpenAI API key                                                       |
| PINECONE_API_KEY                                       | Pinecone API key                                                     |
| PINECONE_INDEX_NAME                                    | Name of your Pinecone index                                          |
| AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION | AWS credentials with read access to your S3 bucket                   |
| S3_BUCKET_NAME                                         | Name of the S3 bucket containing data                   |

3. Create an S3 bucket and upload the PDF files you want to be searchable.
4. Create a Pinecone index:
- Custom settings (this project computes embeddings itself via OpenAI)
- Dimension: 1536 
- Metric: cosine
- Type: Serverless
5. Ingest your documents (can be implemented with github actions)
python src/ingest.py
6. Run the chat app
streamlit run app.py
