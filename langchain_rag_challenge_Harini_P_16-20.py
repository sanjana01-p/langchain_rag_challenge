import os
from dotenv import load_dotenv

load_dotenv(override=True) 


RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

# ─────────────────────────────────────────────────────────────
# TASK 16 — Conversational RAG with Chat History
# ─────────────────────────────────────────────────────────────
"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.

Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]

HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage

  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    # Convert input documents → LangChain Documents
    docs = [Document(page_content=d) for d in documents]

    # Create embeddings + vector store (FAISS → no DB needed)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Step 1: Convert follow-up → standalone question
    contextualize_prompt = ChatPromptTemplate.from_template(
        "Given the chat history and a follow-up question, rewrite it as a standalone question.\n\n"
        "Chat History:\n{chat_history}\n\nQuestion: {input}"
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Step 2: Answer using retrieved context
    qa_prompt = ChatPromptTemplate.from_template(
        "Answer ONLY using the provided context:\n\n{context}\n\nQuestion: {input}"
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    # Chat history
    chat_history = []

    # Turn 1
    response1 = rag_chain.invoke({
        "input": "What is LangChain?",
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content="What is LangChain?"))
    chat_history.append(AIMessage(content=response1["answer"]))

    # Turn 2 (follow-up)
    response2 = rag_chain.invoke({
        "input": "What version introduced LCEL?",
        "chat_history": chat_history
    })

    return [response1["answer"], response2["answer"]]


    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 17 — RAG Agent (Tool-based Retrieval)
# ─────────────────────────────────────────────────────────────
"""
TASK 17: RAG Agent with Retriever as Tool
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.

Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.

HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    # # ── YOUR CODE BELOW ──────────────────────────────────────
    docs = [Document(page_content=d) for d in RAG_DOCUMENTS]

    # Vector store (FAISS → no DB needed)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    # Convert retriever → tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Search technical knowledge base for LangChain, RAG, pgvector info."
    )

    tools = [retriever_tool]

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # ReAct prompt
    prompt = hub.pull("hwchase17/react")

    # Create agent
    agent = create_react_agent(llm, tools, prompt)

    # Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False
    )

    # Run agent
    result = agent_executor.invoke({"input": question})

    return result["output"]
    

    

    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SECTION E — LangSmith  (Tasks 18 – 20)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TASK 18 — Trace a Chain with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 18: LangSmith Tracing
-----------------------------
Instrument a simple LCEL chain so every invocation is
traced in LangSmith.  Your function should:
  1. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT.
  2. Build the same basic LCEL chain from Task 1.
  3. Add run_name and tags to the invocation config.
  4. Return the response AND the run_id of the trace.

Expected return:
  {"answer": str, "run_id": str}

HINT:
  from langchain_core.tracers.context import collect_runs

  with collect_runs() as cb:
      result = chain.invoke(
          {"topic": topic},
          config={"run_name": "task18_trace", "tags": ["challenge"]}
      )
  run_id = str(cb.traced_runs[0].id)
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import collect_runs

def traced_chain(topic: str) -> dict:
    """Runs a chain with LangSmith tracing. Returns answer and run_id."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in simple terms."
    )

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Chain
    chain = prompt | llm | StrOutputParser()

    # Enable tracing
    with collect_runs() as cb:
        result = chain.invoke(
            {"topic": topic},
            config={
                "run_name": "task18_trace",
                "tags": ["assignment", "rag"]
            }
        )

    # Get run_id from LangSmith
    run_id = str(cb.traced_runs[0].id)

    return {
        "answer": result,
        "run_id": run_id
    }



    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 19 — Create a LangSmith Dataset
# ─────────────────────────────────────────────────────────────
"""
TASK 19: Create a LangSmith Dataset and Add Examples
------------------------------------------------------
Use the LangSmith SDK to:
  1. Create a dataset named "rag-eval-dataset".
  2. Add 3 question-answer example pairs to it.
  3. Return the dataset id as a string.

Examples to add:
  Q: "What does RAG stand for?"
     A: "Retrieval-Augmented Generation"
  Q: "What PostgreSQL extension enables vector search?"
     A: "pgvector"
  Q: "What LangChain tool provides observability?"
     A: "LangSmith"

HINT:
  from langsmith import Client
  client = Client()

  dataset = client.create_dataset("rag-eval-dataset")
  client.create_examples(
      inputs=[{"question": q} for q in questions],
      outputs=[{"answer": a} for a in answers],
      dataset_id=dataset.id
  )
"""
from langsmith import Client
def create_langsmith_dataset() -> str:
    """Creates a LangSmith dataset with 3 examples. Returns dataset id."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    client = Client()

    # Create dataset
    dataset = client.create_dataset("rag-eval-dataset")

    # Add examples
    client.create_examples(
        inputs=[
            {"question": "What does RAG stand for?"},
            {"question": "What PostgreSQL extension enables vector search?"},
            {"question": "What LangChain tool provides observability?"}
        ],
        outputs=[
            {"answer": "Retrieval-Augmented Generation"},
            {"answer": "pgvector"},
            {"answer": "LangSmith"}
        ],
        dataset_id=dataset.id
    )

    return str(dataset.id)


    # ── END OF YOUR CODE ─────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TASK 20 — Run an Evaluation with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 20: LangSmith Evaluation (evaluate)
------------------------------------------
Run an automated evaluation of your RAG pipeline using the
dataset created in Task 19.

Steps:
  1. Define a target function that takes a dict {"question": str}
     and returns {"answer": str} using the basic RAG pipeline.
  2. Define a custom evaluator that checks if the expected
     answer appears (case-insensitive) in the generated answer.
  3. Run the evaluation using langsmith.evaluate().
  4. Return the evaluation results summary dict:
     {"dataset": str, "num_examples": int, "pass_rate": float}

HINT:
  from langsmith.evaluation import evaluate, LangChainStringEvaluator

  def target(inputs: dict) -> dict:
      return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}

  results = evaluate(
      target,
      data="rag-eval-dataset",
      evaluators=[...],
      experiment_prefix="rag-challenge-eval",
  )
"""
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    def basic_rag(question):
        docs = [
            "LangChain v0.2 introduced LCEL.",
            "pgvector supports L2, cosine, inner product.",
            "LangSmith is for tracing LLM apps.",
            "RAG means Retrieval-Augmented Generation."
        ]

        documents = [Document(page_content=d) for d in docs]

        embeddings = OpenAIEmbeddings()
        store = FAISS.from_documents(documents, embeddings)

        retriever = store.as_retriever()

        retrieved_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = ChatPromptTemplate.from_template(
            "Answer using context:\n{context}\n\nQuestion: {question}"
        )

        llm = ChatOpenAI(model="gpt-4o-mini")

        return (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })

    # Target function
    def target(inputs):
        return {"answer": basic_rag(inputs["question"])}

    # Evaluator
    def evaluator(run, example):
        predicted = run.outputs["answer"].lower()
        expected = example.outputs["answer"].lower()

        return {"score": 1 if expected in predicted else 0}

    # Run evaluation
    results = evaluate(
        target,
        data="rag-eval-dataset",
        evaluators=[evaluator],
        experiment_prefix="rag-eval"
    )

    # Compute summary
    scores = [r["evaluation_results"][0]["score"] for r in results]

    return {
        "dataset": "rag-eval-dataset",
        "num_examples": len(scores),
        "pass_rate": sum(scores) / len(scores)
    }



    # ── END OF YOUR CODE ─────────────────────────────────────



# =============================================================
#  MAIN — run and print results for each task
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LANGCHAIN · RAG · PGVECTOR · EMBEDDINGS · LANGSMITH")
    print("20-Task Coding Challenge")
    print("=" * 60)

    # ── Section A ─────────────────────────────────────────────
    print("\n── SECTION A: LangChain Core ──────────────────────────\n")


    print("\n[Task 16] Conversational RAG")
    conv_answers = conversational_rag(RAG_DOCUMENTS)
    print("  Turn 1:", conv_answers[0][:80] if conv_answers else "")
    print("  Turn 2:", conv_answers[1][:80] if len(conv_answers) > 1 else "")

    print("\n[Task 17] RAG Agent")
    agent_ans = rag_agent("What distance metrics does pgvector support?")
    print(" ", agent_ans)

    # ── Section E ─────────────────────────────────────────────
    print("\n── SECTION E: LangSmith ───────────────────────────────\n")

    print("[Task 18] Traced Chain")
    traced = traced_chain("embeddings")
    print(f"  Answer : {str(traced.get('answer', ''))[:80]}")
    print(f"  Run ID : {traced.get('run_id')}")

    print("\n[Task 19] Create LangSmith Dataset")
    dataset_id = create_langsmith_dataset()
    print(f"  Dataset ID: {dataset_id}")

    print("\n[Task 20] Run LangSmith Evaluation")
    eval_summary = run_langsmith_evaluation()
    print(f"  Dataset     : {eval_summary.get('dataset')}")
    print(f"  # Examples  : {eval_summary.get('num_examples')}")
    print(f"  Pass rate   : {eval_summary.get('pass_rate')}")

    print("\n" + "=" * 60)
    print("All tasks complete!")
    print("=" * 60)
