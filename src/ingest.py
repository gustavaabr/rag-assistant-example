import os
import logging
import tempfile
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_and_split_pdf(s3, bucket_name, file_key, text_splitter):
    """Download a single PDF from S3, load it, and split it into chunks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, os.path.basename(file_key))
        s3.download_file(bucket_name, file_key, tmp_path)

        docs = PyPDFLoader(tmp_path).load()
        for doc in docs:
            doc.metadata["source"] = file_key

        return text_splitter.split_documents(docs)

def ingest_from_s3():
    bucket_name = os.getenv("S3_BUCKET_NAME")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # boto3 picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION from the environment automatically
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    logger.info("Connecting to S3 bucket: '%s'", bucket_name)

    response = s3.list_objects_v2(Bucket=bucket_name)
    if "Contents" not in response:
        logger.info("Bucket is empty, nothing to ingest.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if not file_key.lower().endswith(".pdf"):
            continue

        logger.info("Processing: %s", file_key)
        try:
            chunks = load_and_split_pdf(s3, bucket_name, file_key, text_splitter)
            all_chunks.extend(chunks)
        except Exception:
            logger.exception("Failed to process '%s', skipping.", file_key)

    if not all_chunks:
        logger.info("No PDF files were successfully processed.")
        return

    logger.info(
        "Split files into %d chunks. Uploading to Pinecone index '%s'...",
        len(all_chunks), index_name)

    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(
          index_name=index_name,
          embedding=embeddings,
          pinecone_api_key=os.getenv("PINECONE_API_KEY"),
      )
    vector_store.add_documents(all_chunks)

    logger.info("Done. All files from S3 are now indexed in Pinecone.")

if __name__ == "__main__":
    ingest_from_s3()