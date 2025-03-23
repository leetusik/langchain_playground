"""Load posts from CSV, clean up, split, ingest into Pinecone for RAG chatbot."""

import logging
import os
from typing import List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    """
    Returns an embedding model instance.
    The chunk_size parameter here is for API batching, not text splitting.
    """
    return OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=200)


def load_posts_from_csv(csv_path: str = "posts.csv") -> List[Document]:
    """
    Load posts from CSV file and convert to documents.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of Documents
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} posts from {csv_path}")

        documents = []
        for _, row in df.iterrows():
            # Only use the content for the document text, title is in metadata
            text = row["content"]

            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "title": row["title"],
                    "author": (
                        row["author"]
                        if "author" in row and pd.notna(row["author"])
                        else ""
                    ),
                    "category": (
                        row["category"]
                        if "category" in row and pd.notna(row["category"])
                        else ""
                    ),
                    "published_date": (
                        row["published_date"]
                        if "published_date" in row and pd.notna(row["published_date"])
                        else ""
                    ),
                    "url": row["url"] if "url" in row and pd.notna(row["url"]) else "",
                },
            )
            documents.append(doc)

        return documents
    except Exception as e:
        logger.error(f"Error loading posts from CSV: {e}")
        return []


def ingest_docs():
    """
    Load posts from CSV, split into chunks, and ingest into Pinecone.
    """
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
    PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

    # Create text splitter optimized for Korean content
    # Korean has higher semantic density per character compared to English
    # For character-based splitting, we need more characters to capture similar semantic content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,  # Increased for Korean content to capture semantically complete ideas
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ".", "? ", "! ", "？", "！", " ", ""],
        keep_separator=False,
    )

    # Get embedding model
    embedding = get_embeddings_model()

    # Initialize Pinecone with new API (v6.0.0+)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, create if it doesn't
    index_list = pc.list_indexes()
    if PINECONE_INDEX_NAME not in [index.name for index in index_list.indexes]:
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,  # Dimension for text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")

    # Load posts
    raw_docs = load_posts_from_csv()

    if not raw_docs:
        logger.error("No documents loaded from CSV")
        return

    logger.info(f"Loaded {len(raw_docs)} documents from posts.csv")

    # Process each document to ensure title is preserved in each chunk
    all_chunks = []
    for doc in raw_docs:
        # Create a temporary document with the content for chunking
        temp_doc = Document(page_content=doc.page_content, metadata=doc.metadata)

        # Split the content into chunks
        content_chunks = text_splitter.split_documents([temp_doc])

        # Add each chunk to the final list
        all_chunks.extend(content_chunks)

    logger.info(
        f"Split into {len(all_chunks)} chunks, each preserving the post metadata"
    )

    # Create vectorstore and directly add documents
    vectorstore = LangchainPinecone.from_documents(
        documents=all_chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME,
        text_key="text",
    )

    logger.info(f"Successfully added {len(all_chunks)} document chunks to Pinecone")

    # Get stats from Pinecone with new API
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    logger.info(f"Vector store now has {stats.total_vector_count} vectors")

    # Update vectorized flag in CSV (optional)
    try:
        df = pd.read_csv("posts.csv")
        df["vectorized"] = True
        df.to_csv("posts.csv", index=False)
        logger.info("Updated vectorized flag in posts.csv")
    except Exception as e:
        logger.warning(f"Could not update vectorized flag in CSV: {e}")


if __name__ == "__main__":
    ingest_docs()
