# Mini RAG — Model Registry para Bioprocesos

**Estado del proyecto:** Activo  
**Curso:** Procesamiento de Lenguaje Natural — Universidad Icesi  
**Programa:** Maestría en Inteligencia Artificial Aplicada

---

## Equipo

| Rol | Nombre | Contacto |
|-----|--------|----------|
| Desarrollador | Álvaro J. Cabrera | [@alvarojcabrera](https://github.com/alvarojcabrera) |

---

## Descripción General

Sistema de **Retrieval-Augmented Generation (RAG)** 100% local y gratuito que permite consultar en lenguaje natural:

- Metadatos de modelos de ML registrados en un **Model Registry** para bioprocesos (producción de penicilina)
- Contenido del paper **STAMM** (Soft sensor moniToring and mAintenance framework for Machine learning Models)
- Resultados de la **evaluación comparativa** de LLMs (Gemini, Meta AI, GPT-5.5) sobre el RAG

### ¿Por qué es diferente al notebook guía del curso?

| Aspecto | Notebook guía (WikiHow) | Este proyecto |
|---|---|---|
| Dataset | Artículos de WikiHow en español | YAMLs de modelos ML + paper científico + evaluación LLMs |
| Chunking | Por palabras (100 words, sliding window) | Semántico por secciones (YAML) + por palabras (PDF/DOCX) |
| Embeddings | sentence-transformers (PyTorch) | Ollama (local, sin PyTorch) |
| Vector store | NumPy array manual | ChromaDB (persistente) |
| LLM | Llama 3 | Qwen 3.5 9B / 4B |
| Costo | $0 | $0 |

---

## Detalles Técnicos

### Métodos y tecnologías

- **RAG** (Retrieval-Augmented Generation)
- **Embeddings** semánticos con similitud coseno
- **Chunking** semántico y por ventana deslizante
- **Historial de conversación** con memoria de 4 interacciones

### Stack tecnológico

| Componente | Herramienta | Versión |
|-----------|-------------|---------|
| LLM (completo) | Qwen 3.5 9B via Ollama | 9B params |
| LLM (ligero) | Qwen 3.5 4B via Ollama | 4B params |
| Embeddings | nomic-embed-text-v2-moe via Ollama | local |
| Vector Store | ChromaDB | ≥0.5.0 |
| Interfaz web | Gradio | ≥6.0 |
| Lectura PDF | PyPDF2 | ≥3.0 |
| Lectura DOCX | python-docx | ≥1.0 |
| Lectura XLSX | openpyxl | ≥3.1 |
| Lenguaje | Python | 3.11 |

### Fuentes de datos ingestadas

| Archivo | Tipo | Chunks | Método |
|---------|------|--------|--------|
| 10 configs de modelos ML | YAML | 40 | Semántico: 4 secciones/modelo |
| 2 project_info | YAML | 2 | 1 chunk/proyecto |
| `stamm_paper.pdf` | PDF | ~109 | 200 palabras + overlap 50 |
| `Manual_Evaluacion_RAG_IndPenSim_PRO.docx` | DOCX | ~10 | 150 palabras + overlap 30 |
| `Rubrica de calificación de los tres modelos.xlsx` | XLSX | ~17 | Por pregunta/respuesta |
| **Total** | | **~178** | |

---

## Cómo instalar y correr

### Prerequisitos

1. **Python 3.11** — [descargar](https://www.python.org/downloads/)
2. **Ollama** — [descargar](https://ollama.com/)

### 1. Clonar el repositorio

```bash
git clone https://github.com/alvarojcabrera/model-registry-rag.git
cd model-registry-rag
```

### 2. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 3. Descargar modelos de Ollama

```bash
# LLM principal (6.6 GB)
ollama pull qwen3.5:9b

# LLM ligero opcional (2.5 GB)
ollama pull qwen3.5:4b

# Modelo de embeddings
ollama pull nomic-embed-text-v2-moe
```

### 4. Verificar que Ollama está corriendo

```bash
ollama list
# Debe mostrar los modelos descargados
```

### 5. Correr la aplicación

**Modo completo — Gradio + Qwen 9B (recomendado):**
```bash
python app.py
# Abre http://127.0.0.1:7860 en tu navegador
```

**Modo ligero — Terminal + Qwen 4B:**
```bash
python app.py --terminal
```

**Si agregas nuevos documentos a `data/`, fuerza re-ingesta:**
```bash
python app.py --reingest
```

> **Nota Windows (PowerShell):** usa `& "C:\ruta\a\python.exe" app.py`

---

## Comandos especiales en el chat

| Comando | Efecto |
|---------|--------|
| Pregunta normal | Responde sin mostrar fuentes |
| Pregunta + `/reference` | Responde mostrando fuentes y score de similitud |

**Ejemplo:**
```
¿Qué es STAMM?                     → Responde directamente
¿Qué es STAMM? /reference          → Responde + muestra de qué archivos viene la info
```

---

## Estructura del repositorio

```
model-registry-rag/
├── app.py                    ← Pipeline RAG completo (ingesta + chatbot + Gradio/Terminal)
├── requirements.txt          ← Dependencias Python
├── .gitignore
├── README.md                 ← Este archivo
│
├── data/                     ← Base de conocimiento
│   ├── stamm_paper.pdf       ← Paper STAMM (58 páginas)
│   ├── Manual_Evaluacion_RAG_IndPenSim_PRO.docx
│   ├── Rubrica de calificación de los tres modelos.xlsx
│   └── projects/
│       ├── Project_IndPenSim/
│       │   ├── project_info.yaml
│       │   └── configs/      ← 10 YAMLs de modelos ML
│       └── Bioindustry_E.Coli/
│           └── project_info.yaml
│
├── docs/                     ← Documentación
│   ├── GUIA_RAG.md           ← Guía detallada del sistema RAG
│   └── EXPLICACION_RAG_LLM.md ← Explicación técnica: RAG + LLMs + evaluación
│
└── notebooks/                ← Versión notebook (para Colab o Jupyter)
    ├── rag_model_registry.ipynb  ← Notebook completo con análisis
    └── Collab.ipynb              ← Versión optimizada para Google Colab
```

---

## Entregables

| Entregable | Descripción | Enlace |
|-----------|-------------|--------|
| `app.py` | Aplicación RAG completa | [ver](app.py) |
| `notebooks/rag_model_registry.ipynb` | Notebook con análisis y plots | [ver](notebooks/rag_model_registry.ipynb) |
| `notebooks/Collab.ipynb` | Versión Google Colab | [ver](notebooks/Collab.ipynb) |
| `docs/GUIA_RAG.md` | Guía paso a paso del RAG | [ver](docs/GUIA_RAG.md) |
| `docs/EXPLICACION_RAG_LLM.md` | Explicación técnica completa | [ver](docs/EXPLICACION_RAG_LLM.md) |

---

## Preguntas de ejemplo

**Sobre los modelos registrados:**
- ¿Qué modelos hay registrados?
- ¿Qué modelos usan Python vs R?
- ¿Cuáles son los inputs del modelo LSTM?
- ¿Qué hiperparámetros usa el SVM?

**Sobre el paper STAMM:**
- ¿Qué es STAMM?
- ¿Quiénes crearon el framework STAMM?

**Sobre la evaluación de LLMs:**
- ¿Qué puntaje obtuvo Gemini en la evaluación?
- ¿Cuáles fueron las preguntas trampa?
- ¿Qué criterios se usaron para evaluar los LLMs?

---

## Referencias

- STAMM Paper: [SSRN 6054948](https://ssrn.com/abstract=6054948)
- Ollama: https://ollama.com
- ChromaDB: https://docs.trychroma.com
- Qwen 3.5: https://huggingface.co/Qwen
- Basado en el notebook guía del curso: [1-ollama-rag](https://github.com/Ohtar10/icesi-nlp/blob/main/Sesion6/1-ollama-rag.ipynb)
