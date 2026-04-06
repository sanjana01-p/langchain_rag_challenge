import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

 

# ─────────────────────────────────────────────────────────────
# TASK 6 — Cosine Similarity (from scratch, then with numpy)
# ─────────────────────────────────────────────────────────────
"""
TASK 6: Cosine Similarity
---------------------------
Part A: Implement cosine_similarity_manual(v1, v2) WITHOUT
        using numpy.  Use only Python loops / math.
Part B: Implement cosine_similarity_numpy(v1, v2) using numpy.

Both should return a float between -1 and 1.

Then embed these two pairs and print which pair is more similar:
  Pair 1: "dog" vs "puppy"
  Pair 2: "dog" vs "automobile"

Formula:
  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)

HINT:
  dot product: sum(a*b for a, b in zip(v1, v2))
  magnitude  : sum(x**2 for x in v) ** 0.5
  numpy equiv: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
"""

def cosine_similarity_manual(v1: list, v2: list) -> float:
    """Computes cosine similarity using pure Python."""
    import math

    # Dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Magnitudes
    magnitude_v1 = math.sqrt(sum(x ** 2 for x in v1))
    magnitude_v2 = math.sqrt(sum(x ** 2 for x in v2))
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    
    return dot_product / (magnitude_v1 * magnitude_v2)

def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """Computes cosine similarity using numpy."""
    import numpy as np
    
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    numerator = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

from langchain_openai import OpenAIEmbeddings

def compare_word_pairs() -> dict:
    """
    Embeds dog/puppy and dog/automobile, returns:
    {
      "dog_vs_puppy"      : float,
      "dog_vs_automobile" : float,
      "more_similar_pair" : str
    }
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    dog_vec = embeddings.embed_query("dog")
    puppy_vec = embeddings.embed_query("puppy")
    auto_vec = embeddings.embed_query("automobile")
    
    sim_dog_puppy = cosine_similarity_numpy(dog_vec, puppy_vec)
    sim_dog_auto = cosine_similarity_numpy(dog_vec, auto_vec)
    
    if sim_dog_puppy > sim_dog_auto:
        more_similar = "dog vs puppy"
    else:
        more_similar = "dog vs automobile"
    
    return {
        "dog_vs_puppy": sim_dog_puppy,
        "dog_vs_automobile": sim_dog_auto,
        "more_similar_pair": more_similar
    }


# ─────────────────────────────────────────────────────────────
# TASK 7 — Batch Embedding with Chunking
# ─────────────────────────────────────────────────────────────
"""
TASK 7: Batch Embedding with Chunking
----------------------------------------
Given a long text document, split it into overlapping chunks
using RecursiveCharacterTextSplitter, then embed all chunks
in a single batch call.  Return:
  {
    "num_chunks"   : int,
    "chunk_size"   : int,   # configured chunk size
    "overlap"      : int,   # configured overlap
    "embedding_dim": int,
    "chunks"       : list[str]
  }

Use chunk_size=200, chunk_overlap=40.

HINT:
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=200, chunk_overlap=40
  )
  chunks = splitter.split_text(long_text)
  vectors = embeddings.embed_documents(chunks)
"""

SAMPLE_DOCUMENT = """
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, agents, and memory.
LangChain integrates with many LLM providers including OpenAI, Anthropic, and Cohere.
The framework also supports vector stores, document loaders, and output parsers.
RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by fetching relevant documents from a knowledge base at query time.
pgvector is a PostgreSQL extension that enables efficient storage and similarity
search of high-dimensional vector embeddings directly inside a relational database.
LangSmith is an observability platform for LangChain applications that provides
tracing, evaluation, and debugging of LLM pipelines.
"""


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def batch_embed_with_chunks(text: str, chunk_size: int, overlap: int) -> dict:
    """Splits text into chunks, embeds them, and returns metadata."""
    
    # 1. Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    # 2. Split text into chunks
    chunks = splitter.split_text(text)
    
    # 3. Initialize embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 4. Batch embed all chunks
    vectors = embeddings.embed_documents(chunks)
    
    # 5. Get embedding dimension
    embedding_dim = len(vectors[0]) if vectors else 0
    
    # 6. Return metadata
    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": embedding_dim,
        "chunks": chunks
    }


# ─────────────────────────────────────────────────────────────
# TASK 8 — Compare Two Embedding Models
# ─────────────────────────────────────────────────────────────
"""
TASK 8: Compare Two Embedding Models
--------------------------------------
Embed the same sentence using two different OpenAI models:
  Model A: text-embedding-3-small   (1536 dims)
  Model B: text-embedding-3-large   (3072 dims)

For the sentence:  "Vector databases power semantic search."

Return a dict:
  {
    "sentence"   : str,
    "model_a"    : {"model": str, "dims": int, "first_3": list[float]},
    "model_b"    : {"model": str, "dims": int, "first_3": list[float]},
    "dim_ratio"  : float   # model_b_dims / model_a_dims
  }

HINT:
  OpenAIEmbeddings(model="text-embedding-3-small")
  OpenAIEmbeddings(model="text-embedding-3-large")
  embeddings.embed_query(sentence) → single vector (list of floats)
"""


from langchain_openai import OpenAIEmbeddings

def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    
    # Model A: small
    model_a = OpenAIEmbeddings(model="text-embedding-3-small")
    vec_a = model_a.embed_query(sentence)
    
    # Model B: large
    model_b = OpenAIEmbeddings(model="text-embedding-3-large")
    vec_b = model_b.embed_query(sentence)
    
    # Dimensions
    dim_a = len(vec_a)
    dim_b = len(vec_b)
    
    return {
        "sentence": sentence,
        "model_a": {
            "model": "text-embedding-3-small",
            "dims": dim_a,
            "first_3": vec_a[:3]
        },
        "model_b": {
            "model": "text-embedding-3-large",
            "dims": dim_b,
            "first_3": vec_b[:3]
        },
        "dim_ratio": dim_b / dim_a if dim_a != 0 else 0
    }



# ─────────────────────────────────────────────────────────────
# SECTION D — RAG Agents  (Tasks 14 – 17)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 14 — Basic RAG Pipeline
# ─────────────────────────────────────────────────────────────
"""
TASK 14: Basic RAG Pipeline
------------------------------
Build an end-to-end RAG chain that:
  1. Loads documents from a list of strings.
  2. Stores them in a PGVector vectorstore.
  3. Creates a retriever (top-3 results).
  4. Passes retrieved context + question to ChatOpenAI.
  5. Returns the final answer string.

Use the LCEL pattern:
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

HINT:
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  prompt = ChatPromptTemplate.from_template(
      "Answer using only this context:\n{context}\n\nQuestion: {question}"
  )
"""

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def basic_rag_pipeline(documents: list, question: str) -> str:
    """Indexes documents and answers the question using RAG."""
    
    # 1. Convert to Document objects
    docs = [Document(page_content=d) for d in documents]
    
    # 2. Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 3. Vector store (PGVector)
    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        connection="postgresql+psycopg://postgres:postgres123@localhost:5432/vectordb",
        collection_name="rag_collection"
    )
    
    # 4. Retriever (top-3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 5. Prompt
    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )
    
    # 6. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 7. LCEL Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 8. Run
    return chain.invoke(question)


# ─────────────────────────────────────────────────────────────
# TASK 15 — RAG with Source Attribution
# ─────────────────────────────────────────────────────────────
"""
TASK 15: RAG with Source Attribution
---------------------------------------
Extend the RAG pipeline to also return the source documents
used to generate the answer.  Return a dict:
  {
    "answer" : str,
    "sources": [{"content": str, "score": float}, ...]
  }

HINT:
  Use RunnableParallel to run retrieval and generation
  in parallel, or retrieve docs first and pass them to both
  the formatter and the chain:

  from langchain_core.runnables import RunnableParallel, RunnablePassthrough

  retrieval_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )
  # Then use the context in both the answer chain and as sources.
"""

from langchain_core.runnables import RunnableParallel


def rag_with_sources(documents: list, question: str) -> dict:
    """Returns the answer AND the source documents used."""
    
    # 1. Convert to Document objects
    docs = [Document(page_content=d) for d in documents]
    
    # 2. Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 3. Vector store
    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        connection="postgresql+psycopg://postgres:postgres123@localhost:5432/vectordb",
        collection_name="rag_collection_sources"
    )
    
    # 4. Retriever (top-3 with scores)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 5. Prompt
    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )
    
    # 6. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 7. Retrieve docs first
    retrieved_docs = retriever.invoke(question)
    
    # 8. Format context
    context = format_docs(retrieved_docs)
    
    # 9. Generate answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    # 10. Prepare sources (no score directly → mock similarity)
    sources = [
        {"content": doc.page_content, "score": 1.0}  # PGVector retriever doesn't return score by default
        for doc in retrieved_docs
    ]
    
    return {
        "answer": answer,
        "sources": sources
    }



if __name__ == "__main__":

    print("=" * 60)
    print("LANGCHAIN · RAG · PGVECTOR · EMBEDDINGS · LANGSMITH")
    print("20-Task Coding Challenge")
    print("=" * 60)


    print("\n[Task 6] Cosine Similarity")
    word_pairs = compare_word_pairs()
    print(f"  dog vs puppy      : {word_pairs.get('dog_vs_puppy', ''):.4f}")
    print(f"  dog vs automobile : {word_pairs.get('dog_vs_automobile', ''):.4f}")
    print(f"  More similar      : {word_pairs.get('more_similar_pair')}")

    print("\n[Task 7] Batch Embedding with Chunking")
    chunk_info = batch_embed_with_chunks(SAMPLE_DOCUMENT, 200, 40)
    print(f"  Chunks     : {chunk_info.get('num_chunks')}")
    print(f"  Embed dims : {chunk_info.get('embedding_dim')}")

    print("\n[Task 8] Compare Embedding Models")
    model_cmp = compare_embedding_models("Vector databases power semantic search.")
    print(f"  Model A dims : {model_cmp.get('model_a', {}).get('dims')}")
    print(f"  Model B dims : {model_cmp.get('model_b', {}).get('dims')}")
    print(f"  Dim ratio    : {model_cmp.get('dim_ratio')}")

    
    # ── Section D ─────────────────────────────────────────────
    print("\n── SECTION D: RAG Agents ──────────────────────────────\n")

    print("[Task 14] Basic RAG Pipeline")
    rag_ans = basic_rag_pipeline(RAG_DOCUMENTS, "What is LCEL?")
    print(" ", rag_ans)

    print("\n[Task 15] RAG with Source Attribution")
    rag_src = rag_with_sources(RAG_DOCUMENTS, "What distance metrics does pgvector support?")
    print("  Answer  :", rag_src.get("answer", ""))
    print("  Sources :")
    for s in rag_src.get("sources", []):
        print(f"    [{s.get('score', 0):.4f}] {s.get('content', '')[:60]}")
