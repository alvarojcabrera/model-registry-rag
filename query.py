# -*- coding: utf-8 -*-
"""
Mini RAG - Interfaz de chat para consultar metadatos del model registry.
Usa ChromaDB para retrieval y Qwen 3.5 9B via Ollama para generación.
"""

import os
import chromadb
import requests

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text-v2-moe"
LLM_MODEL = "qwen3.5:9b"
TOP_K = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

SYSTEM_PROMPT = """Eres un asistente experto en el registro de modelos de machine learning para bioprocesos.
Respondes preguntas sobre los modelos registrados, sus configuraciones, inputs, outputs, hiperparámetros y proyectos.
Usa SOLO la información del contexto proporcionado para responder. Si no encuentras la respuesta en el contexto, dilo claramente.
Responde de forma concisa y precisa. Puedes responder en español o inglés según la pregunta del usuario."""


def ollama_embed(text: str) -> list[float]:
    """Genera embedding para un texto."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def ollama_generate(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Genera respuesta con el LLM via Ollama (streaming)."""
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

    full_response = ""
    for line in resp.iter_lines():
        if line:
            import json
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            full_response += token
    print()
    return full_response


def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    """Busca los chunks más relevantes en ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("model_registry")

    query_embedding = ollama_embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results["documents"][0]


def ask(question: str) -> str:
    """Pipeline RAG completo: retrieve + generate."""
    # Retrieve
    context_docs = retrieve(question)
    context = "\n\n---\n\n".join(context_docs)

    # Build prompt
    prompt = f"""Contexto del registro de modelos:

{context}

---

Pregunta del usuario: {question}

Respuesta:"""

    # Generate
    return ollama_generate(prompt)


def chat():
    """Loop interactivo de chat."""
    print("=" * 60)
    print("  Mini RAG - Model Registry Assistant")
    print("  Escribe tu pregunta o 'salir' para terminar")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("Tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit"):
            print("Hasta luego!")
            break

        print("\nAsistente > ", end="")
        ask(question)
        print()


if __name__ == "__main__":
    chat()
