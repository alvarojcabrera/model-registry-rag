# -*- coding: utf-8 -*-
"""
Ingesta de YAMLs del model-registry hacia ChromaDB.
Parsea configs de modelos y project_info, genera chunks y los almacena.
"""

import os
import yaml
import chromadb
import requests

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text-v2-moe"

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(SCRIPT_DIR, "..", "projects")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")


def ollama_embed(texts: list[str]) -> list[list[float]]:
    """Genera embeddings usando Ollama."""
    embeddings = []
    for text in texts:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embeddings"][0])
    return embeddings


def yaml_to_chunks(yaml_path: str) -> list[dict]:
    """Convierte un YAML de config de modelo en chunks de texto legible."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chunks = []
    filename = os.path.basename(yaml_path)

    # --- Si es un project_info.yaml ---
    if "project_ID" in data:
        text = f"Proyecto: {data.get('project_name', 'N/A')}\n"
        text += f"ID: {data.get('project_ID', 'N/A')}\n"
        text += f"Descripción: {data.get('description', 'N/A')}\n"
        text += f"Coordinador: {data.get('coordinator', 'N/A')}\n"
        text += f"Periodo: {data.get('start_date', '')} - {data.get('end_date', '')}\n"

        if "references" in data:
            refs = "\n".join(r.get("apa", "") for r in data["references"])
            text += f"Referencias:\n{refs}\n"

        if "variables" in data:
            var_lines = []
            for v in data["variables"]:
                var_lines.append(
                    f"  - {v['renamed_variable']} ({v['units']}): {v['description']}"
                )
            text += f"Variables del proceso:\n" + "\n".join(var_lines)

        chunks.append({
            "id": data["project_ID"],
            "text": text,
            "source": filename,
            "doc_type": "project_info",
        })
        return chunks

    # --- Si es un config de modelo ---
    cfg = data.get("ml_model_configuration", {})
    ident = cfg.get("model_identification", {})
    desc = cfg.get("model_description", {})
    training = cfg.get("training_information", {})
    inputs = cfg.get("inputs", {})
    outputs = cfg.get("outputs", {})

    model_id = ident.get("ID", "unknown")

    # Chunk 1: Identificación y descripción general
    text1 = f"Modelo: {ident.get('name', 'N/A')}\n"
    text1 += f"ID: {model_id}\n"
    text1 += f"Versión: {ident.get('version', 'N/A')}\n"
    text1 += f"Autor: {ident.get('author', 'N/A')}\n"
    text1 += f"DOI: {ident.get('doi', 'N/A')}\n"
    text1 += f"Fecha de creación: {ident.get('creation_date', 'N/A')}\n"
    text1 += f"Estado: {ident.get('status', 'N/A')} - {ident.get('status_description', '')}\n"
    text1 += f"Tipo: {desc.get('model_type', 'N/A')}\n"
    text1 += f"Learner: {desc.get('learner', 'N/A')}\n"
    text1 += f"Nombre completo: {desc.get('model_name', 'N/A')}\n"
    text1 += f"Descripción: {desc.get('description', 'N/A')}\n"

    langs = desc.get("language", [])
    if langs:
        lang_str = ", ".join(
            f"{l.get('name', '')} {l.get('version', '')}".strip() for l in langs
        )
        text1 += f"Lenguaje: {lang_str}\n"

    pkgs = desc.get("packages", [])
    if pkgs:
        pkg_parts = []
        for p in pkgs:
            if isinstance(p, dict):
                name = p.get("package", "")
                cls = p.get("class", "")
                ver = p.get("version", "")
                label = f"{name}.{cls}" if cls else name
                if ver:
                    label += f" v{ver}"
                pkg_parts.append(label)
            else:
                pkg_parts.append(str(p))
        text1 += f"Paquetes: {', '.join(pkg_parts)}\n"

    chunks.append({
        "id": f"{model_id}_overview",
        "text": text1,
        "source": filename,
        "doc_type": "model_overview",
    })

    # Chunk 2: Training info
    if training:
        text2 = f"Entrenamiento del modelo {ident.get('name', '')} (ID: {model_id}):\n"
        text2 += f"Instancias: {training.get('number_of_instances', 'N/A')}\n"
        text2 += f"Validación: {training.get('validation', 'N/A')}\n"

        hyper = training.get("hyperparameters", {})
        if hyper:
            hp_lines = [f"  - {k}: {v}" for k, v in hyper.items()]
            text2 += f"Hiperparámetros:\n" + "\n".join(hp_lines) + "\n"

        chunks.append({
            "id": f"{model_id}_training",
            "text": text2,
            "source": filename,
            "doc_type": "model_training",
        })

    # Chunk 3: Inputs (features)
    features = inputs.get("features", [])
    if features:
        text3 = f"Inputs del modelo {ident.get('name', '')} (ID: {model_id}):\n"
        for feat in features:
            text3 += (
                f"  - {feat['name']} ({feat.get('units', 'N/A')}): "
                f"{feat.get('description', '')} "
                f"[tipo: {feat.get('type', 'N/A')}, "
                f"rango: {feat.get('expected_range', {}).get('min', '?')}"
                f"-{feat.get('expected_range', {}).get('max', '?')}]\n"
            )
        chunks.append({
            "id": f"{model_id}_inputs",
            "text": text3,
            "source": filename,
            "doc_type": "model_inputs",
        })

    # Chunk 4: Outputs
    out_info = outputs.get("information", [])
    if out_info:
        text4 = f"Outputs del modelo {ident.get('name', '')} (ID: {model_id}):\n"
        for out in out_info:
            text4 += (
                f"  - {out['name']} ({out.get('units', 'N/A')}): "
                f"{out.get('description', '')} "
                f"[rango: {out.get('expected_range', {}).get('min', '?')}"
                f"-{out.get('expected_range', {}).get('max', '?')}]\n"
            )
        chunks.append({
            "id": f"{model_id}_outputs",
            "text": text4,
            "source": filename,
            "doc_type": "model_outputs",
        })

    return chunks


def ingest():
    """Recorre todos los proyectos, parsea YAMLs e ingesta en ChromaDB."""
    all_chunks = []

    for project_folder in os.listdir(PROJECTS_DIR):
        project_path = os.path.join(PROJECTS_DIR, project_folder)
        if not os.path.isdir(project_path):
            continue

        # project_info.yaml
        info_file = os.path.join(project_path, "project_info.yaml")
        if os.path.exists(info_file):
            all_chunks.extend(yaml_to_chunks(info_file))

        # configs/*.yaml
        configs_dir = os.path.join(project_path, "configs")
        if os.path.exists(configs_dir):
            for cfg_file in sorted(os.listdir(configs_dir)):
                if cfg_file.endswith(".yaml"):
                    all_chunks.extend(
                        yaml_to_chunks(os.path.join(configs_dir, cfg_file))
                    )

    if not all_chunks:
        print("No se encontraron documentos para ingestar.")
        return

    print(f"Generando embeddings para {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = ollama_embed(texts)

    # Guardar en ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recrear colección limpia
    try:
        client.delete_collection("model_registry")
    except Exception:
        pass

    collection = client.create_collection(
        name="model_registry",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": c["source"], "doc_type": c["doc_type"]}
            for c in all_chunks
        ],
    )

    print(f"Ingesta completa: {len(all_chunks)} chunks en ChromaDB.")
    print("Chunks creados:")
    for c in all_chunks:
        print(f"  [{c['doc_type']}] {c['id']}")


if __name__ == "__main__":
    ingest()
