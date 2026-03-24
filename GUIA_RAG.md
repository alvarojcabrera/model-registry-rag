# Mini RAG - Model Registry: Guía Paso a Paso

## ¿Qué es un RAG?

RAG = **Retrieval-Augmented Generation** (Generación Aumentada por Recuperación).

Es una técnica que combina dos cosas:
1. **Buscar** información relevante en tus documentos
2. **Generar** una respuesta usando un LLM, pero alimentándolo con esa información encontrada

Sin RAG, el LLM solo sabe lo que aprendió en su entrenamiento.
Con RAG, el LLM puede "leer" TUS datos y responder sobre ellos.

```
┌──────────────────────────────────────────────────────┐
│                  FLUJO DE UNA PREGUNTA                │
│                                                       │
│  Usuario: "¿Qué modelos usan Random Forest?"         │
│       │                                               │
│       ▼                                               │
│  [1] Convertir pregunta → vector (embedding)          │
│       │                                               │
│       ▼                                               │
│  [2] Buscar vectores similares en ChromaDB            │
│       │                                               │
│       ▼                                               │
│  [3] Recuperar los 5 chunks más relevantes            │
│       │                                               │
│       ▼                                               │
│  [4] Armar prompt: "Dado este contexto... responde"   │
│       │                                               │
│       ▼                                               │
│  [5] Qwen 3.5 genera la respuesta                     │
│       │                                               │
│       ▼                                               │
│  Respuesta: "Hay 2 modelos RF: 0001 (Python)..."      │
└──────────────────────────────────────────────────────┘
```

---

## Arquitectura: Los 4 Componentes

### Componente 1: Fuente de datos (YAMLs)

**¿Qué son?** Los archivos YAML del model-registry contienen metadatos de cada modelo:
- Nombre, autor, versión, fecha
- Tipo de algoritmo (RF, SVM, LSTM, etc.)
- Inputs (features) con rangos esperados
- Outputs (qué predice)
- Hiperparámetros de entrenamiento

**¿Por qué YAMLs y no PDFs?**
- Ya teníamos datos estructurados y limpios
- No necesitamos OCR ni parseo complejo
- El caso de uso es concreto: consultar metadatos de modelos

### Componente 2: Embeddings (nomic-embed-text-v2-moe)

**¿Qué es un embedding?**
Es convertir texto en un vector numérico (lista de números). Textos con significado
similar producen vectores cercanos entre sí.

```
"Random Forest con 5 árboles" → [0.12, -0.45, 0.78, ...]
"RF usando 5 estimadores"     → [0.11, -0.44, 0.77, ...]  ← ¡Muy similar!
"Temperatura del reactor"     → [0.89, 0.23, -0.56, ...]  ← Diferente
```

**¿Por qué nomic-embed-text-v2-moe?**
- Ya lo tenía instalado en Ollama
- Corre 100% local y gratis
- Buen rendimiento para textos técnicos
- Evita instalar PyTorch (~2GB) que necesitaría sentence-transformers

**Alternativa considerada:** `all-MiniLM-L6-v2` (sentence-transformers)
- Requiere instalar torch + sentence-transformers (~3GB extra)


### Componente 3: Vector Store (ChromaDB)

**¿Qué es?**
Una base de datos especializada en buscar por similitud de vectores.
Cuando haces una pregunta, ChromaDB encuentra los chunks cuyos vectores
son más parecidos al vector de tu pregunta.

**¿Por qué ChromaDB?**
- Instalación simple: `pip install chromadb` y listo
- No requiere servidor externo (es embedded, corre dentro de Python)
- Persiste en disco (carpeta `chroma_db/`)
- API sencilla: add(), query(), y poco más

**Alternativas consideradas:**
- FAISS (Facebook): más rápido para millones de vectores, pero más complejo de usar
- Pinecone/Weaviate: requieren servidor o son servicios cloud (no gratis)
- Para 42 chunks, ChromaDB es más que suficiente

### Componente 4: LLM (Qwen 3.5 9B via Ollama)

**¿Qué hace?**
Recibe el contexto recuperado + la pregunta del usuario y genera una respuesta
en lenguaje natural.

**¿Por qué Ollama?**
- Simplifica correr LLMs locales (descarga, cuantización, servidor API)
- API REST estándar compatible con cualquier cliente
- Gratis y sin telemetría

---

## El Proceso de Ingesta (ingest.py)

### ¿Qué es un "chunk"?

En vez de meter un YAML entero como un solo documento, lo dividimos en pedazos
temáticos (chunks). Esto mejora la precisión del retrieval.

Cada modelo genera **4 chunks**:

| Chunk | Contenido | Por qué separarlo |
|-------|-----------|-------------------|
| `overview` | Nombre, autor, tipo, lenguaje, descripción | Preguntas generales sobre el modelo |
| `training` | Instancias, hiperparámetros, validación | Preguntas sobre entrenamiento |
| `inputs` | Features con tipos, unidades, rangos | Preguntas sobre qué datos necesita |
| `outputs` | Variables de salida, rangos | Preguntas sobre qué predice |

Cada `project_info.yaml` genera **1 chunk** con info del proyecto.

**Total: 10 modelos × 4 chunks + 2 proyectos = 42 chunks**

### Flujo de ingest.py

```
1. Recorrer projects/*/
2. Para cada proyecto:
   a. Leer project_info.yaml → 1 chunk
   b. Leer configs/*.yaml → 4 chunks por modelo
3. Enviar todos los textos a Ollama para generar embeddings
4. Guardar textos + embeddings + metadatos en ChromaDB
```

---

## El Proceso de Consulta (query.py)

### Flujo de query.py

```
1. Usuario escribe pregunta
2. Se genera embedding de la pregunta (Ollama)
3. ChromaDB busca los 5 chunks más similares (TOP_K=5)
4. Se arma un prompt con formato:
   "Contexto: [chunks encontrados]
    Pregunta: [pregunta del usuario]
    Respuesta:"
5. Se envía al LLM (Qwen 3.5) via Ollama
6. Se muestra la respuesta en streaming (token por token)
```

### El System Prompt

Le decimos al LLM:
- Que es experto en el registro de modelos
- Que SOLO use la información del contexto (no invente)
- Que responda conciso y en el idioma de la pregunta

---

## Estructura de Archivos

```
model-registry/
├── rag/
│   ├── ingest.py        ← Parsea YAMLs y los mete en ChromaDB
│   ├── query.py         ← Chat interactivo con RAG
│   ├── chroma_db/       ← Base de datos de vectores (generada)
│   ├── requirements.txt ← Dependencias Python
│   └── GUIA_RAG.md      ← Este archivo
├── projects/
│   ├── Project_IndPenSim/
│   │   ├── project_info.yaml
│   │   ├── configs/     ← 10 YAMLs de modelos
│   │   └── models/      ← Archivos .pkl, .keras, .rds
│   └── Bioindustry_E.Coli/
│       └── project_info.yaml
└── core/
    └── model_registry.py ← Registry original (no modificado)
```

---

## Decisiones de Diseño Resumidas

| Decisión | Elegido | Razón |
|----------|---------|-------|
| Fuente de datos | YAMLs del registry | Datos limpios y estructurados ya existentes |
| Embeddings | nomic-embed-text-v2-moe | Ya instalado en Ollama, 100% local |
| Vector store | ChromaDB | Simple, embedded, suficiente para ~42 chunks |
| LLM | Qwen 3.5 9B | Elegido por el usuario, corre local via Ollama |
| Chunking | 4 chunks por modelo | Mejora precisión vs. 1 chunk gigante por modelo |
| Framework | Python puro | Sin LangChain = menos dependencias, más control |
| Costo | $0 | Todo local, sin APIs externas |
