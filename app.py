# -*- coding: utf-8 -*-
"""
Mini RAG - Streamlit UI for Model Registry Assistant.
Estilo visual consistente con STAMM UI-FastApi.
"""

import streamlit as st
import chromadb
import requests
import json
import os
import yaml

# --- Config ---
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text-v2-moe"
LLM_MODEL = "qwen3.5:9b"
TOP_K = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
PROJECTS_DIR = os.path.join(SCRIPT_DIR, "..", "projects")

SYSTEM_PROMPT = """Eres un asistente experto en el registro de modelos de machine learning para bioprocesos.
Respondes preguntas sobre los modelos registrados, sus configuraciones, inputs, outputs, hiperparámetros y proyectos.
Usa SOLO la información del contexto proporcionado para responder. Si no encuentras la respuesta en el contexto, dilo claramente.
Responde de forma concisa y precisa. Puedes responder en español o inglés según la pregunta del usuario."""

# --- Page Config ---
st.set_page_config(
    page_title="STAMM - RAG Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Ollama Functions ---
def ollama_embed(text: str) -> list:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def ollama_generate_stream(prompt: str, system: str = SYSTEM_PROMPT):
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        },
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token


def retrieve(query: str, top_k: int = TOP_K):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("model_registry")
    query_embedding = ollama_embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results["documents"][0], results["metadatas"][0]


# --- Sidebar ---
st.sidebar.title("🗂️ Project")

# Logo
logo_path = os.path.join(ASSETS_DIR, "ml_repo_logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

# Project info from filesystem
project_map = {}
for folder in os.listdir(PROJECTS_DIR):
    folder_path = os.path.join(PROJECTS_DIR, folder)
    info_file = os.path.join(folder_path, "project_info.yaml")
    if os.path.isdir(folder_path) and os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)
            pid = info.get("project_ID")
            pname = info.get("project_name", folder)
            if pid:
                project_map[pid] = {"name": pname, "info": info}

if project_map:
    project_ids = list(project_map.keys())
    project_names = [project_map[pid]["name"] for pid in project_ids]
    selected_idx = st.sidebar.selectbox(
        "Select a Project:",
        range(len(project_ids)),
        format_func=lambda i: project_names[i],
    )
    selected_project = project_map[project_ids[selected_idx]]

    with st.sidebar.expander("ℹ️ Project Info"):
        st.markdown(f"### {selected_project['name']}")
        info = selected_project["info"]
        st.write(info.get("description", "No description available.")[:300] + "...")
        st.write(f"**Coordinator:** {info.get('coordinator', 'N/A')}")
        st.write(
            f"**Period:** {info.get('start_date', '')} - {info.get('end_date', '')}"
        )

# RAG Settings
st.sidebar.markdown("---")
st.sidebar.title("⚙️ RAG Settings")
top_k = st.sidebar.slider("Chunks to retrieve (TOP_K)", 1, 10, TOP_K)
show_context = st.sidebar.checkbox("Show retrieved context", value=True)

# Model info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**LLM:** `{LLM_MODEL}`")
st.sidebar.markdown(f"**Embeddings:** `{EMBED_MODEL}`")
st.sidebar.markdown(f"**Vector DB:** ChromaDB")

# --- Main Area ---
st.markdown(
    "<h2>🧬 STAMM - RAG Assistant</h2>"
    "<p style='color: gray;'>Ask questions about your registered ML models</p>",
    unsafe_allow_html=True,
)

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    # Show context expander after assistant messages
    if (
        msg["role"] == "assistant"
        and show_context
        and i < len(st.session_state.contexts) * 2
    ):
        ctx_idx = i // 2
        if ctx_idx < len(st.session_state.contexts):
            ctx = st.session_state.contexts[ctx_idx]
            with st.expander(f"📄 Retrieved chunks ({len(ctx['docs'])})"):
                for j, (doc, meta) in enumerate(zip(ctx["docs"], ctx["metas"])):
                    st.markdown(
                        f"**Chunk {j+1}** — `{meta.get('source', '?')}` "
                        f"({meta.get('doc_type', '?')})"
                    )
                    st.code(doc, language=None)

# --- Chat Input ---
if question := st.chat_input("Ask about your models..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve
    with st.spinner("🔍 Searching relevant chunks..."):
        docs, metas = retrieve(question, top_k=top_k)

    # Store context for display
    st.session_state.contexts.append({"docs": docs, "metas": metas})

    # Build prompt
    context = "\n\n---\n\n".join(docs)
    prompt = f"""Contexto del registro de modelos:

{context}

---

Pregunta del usuario: {question}

Respuesta:"""

    # Generate with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for token in ollama_generate_stream(prompt):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    # Show context
    if show_context:
        with st.expander(f"📄 Retrieved chunks ({len(docs)})"):
            for j, (doc, meta) in enumerate(zip(docs, metas)):
                st.markdown(
                    f"**Chunk {j+1}** — `{meta.get('source', '?')}` "
                    f"({meta.get('doc_type', '?')})"
                )
                st.code(doc, language=None)
