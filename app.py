"""Minimal RAG API with FastAPI + LangChain + Gemini.

Key fixes vs. previous version:
- No hard-coded API key; loads from environment/.env.
- Chroma store is persisted and auto-reloaded across restarts.
- Input validation with HTTP errors instead of silent failures.
- Defensive handling when nothing is ingested yet.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Environment -----------------------------------------------------------------
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env with GEMINI_API_KEY=<your-key> or export it."
    )


# --- FastAPI setup ----------------------------------------------------------------
app = FastAPI()


# --- Models -----------------------------------------------------------------------
class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    query: str


# --- LLM + embeddings -------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# --- Vector store -----------------------------------------------------------------
PERSIST_DIR = Path("./chroma_db")
vector_store: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Return an initialized Chroma store (load persisted data if present)."""
    global vector_store

    if vector_store is None:
        vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    return vector_store


# --- Routes -----------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    pdf_path = Path(req.file_path)
    if not pdf_path.is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    store = get_vector_store()
    store.add_documents(splits)
    store.persist()  # make sure data survives restarts

    return {"status": "Documents ingested", "chunks": len(splits)}


@app.post("/query")
def query(req: QueryRequest):
    store = get_vector_store()

    # If no documents ingested yet
    if store._collection.count() == 0:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail="No documents ingested yet")

    results = store.similarity_search(req.query, k=5)
    if not results:
        return {"answer": "I don't know", "sources": []}

    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""
Use ONLY the context below to answer.
If not found, say "I don't know".

Context:
{context}

Question:
{req.query}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in results],
    }

