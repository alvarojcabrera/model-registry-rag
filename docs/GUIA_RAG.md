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

### Componente 1: Fuentes de datos (YAMLs + PDF)

El RAG lee dos tipos de documentos:

**YAMLs del model-registry** — metadatos estructurados de cada modelo:
- Nombre, autor, versión, fecha
- Tipo de algoritmo (RF, SVM, LSTM, etc.)
- Inputs (features) con rangos esperados
- Outputs (qué predice)
- Hiperparámetros de entrenamiento

**PDF del paper STAMM** — documento técnico sobre el framework:
- Descripción del framework STAMM
- Casos de uso y motivación
- Marco teórico de soft sensors y model registries

Los YAMLs se chunkean por secciones temáticas. El PDF se chunkea por palabras
(200 palabras por chunk, con overlap de 50 palabras).

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
- Ya estaba instalado en Ollama
- Corre 100% local y gratis
- Buen rendimiento para textos técnicos
- Evita instalar PyTorch (~2GB) que necesitaría sentence-transformers

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
- Para 151 chunks, ChromaDB es más que suficiente

### Componente 4: LLM (Qwen 3.5 via Ollama)

**Dos modos disponibles:**

| Modo | Modelo | Interfaz | Comando |
|------|--------|----------|---------|
| Completo | Qwen 3.5 9B | Gradio (UI web) | `python app.py` |
| Ligero | Qwen 3.5 4B | Terminal | `python app.py --terminal` |

**¿Por qué Ollama?**
- Simplifica correr LLMs locales (descarga, cuantización, servidor API)
- API REST estándar compatible con cualquier cliente
- Gratis y sin telemetría

**Optimizaciones aplicadas:**
- `think: False` — desactiva el modo de razonamiento interno de Qwen (3-5x más rápido)
- `num_predict: 300` — limita tokens de respuesta para no extenderse

---

## El Proceso de Ingesta (app.py)

### ¿Qué es un "chunk"?

En vez de meter un documento entero, lo dividimos en pedazos temáticos.
Esto mejora la precisión del retrieval porque cada chunk tiene un tema concreto.

### Chunking de YAMLs (semántico por secciones)

Cada modelo genera **4 chunks**:

| Chunk | Contenido | Por qué separarlo |
|-------|-----------|-------------------|
| `overview` | Nombre, autor, tipo, lenguaje, descripción | Preguntas generales sobre el modelo |
| `training` | Instancias, hiperparámetros, validación | Preguntas sobre entrenamiento |
| `inputs` | Features con tipos, unidades, rangos | Preguntas sobre qué datos necesita |
| `outputs` | Variables de salida, rangos | Preguntas sobre qué predice |

Cada `project_info.yaml` genera **1 chunk** con info del proyecto.

**Total YAMLs: 10 modelos × 4 chunks + 2 proyectos = 42 chunks**

### Chunking del PDF (por palabras con sliding window)

El PDF del paper STAMM (58 páginas, ~23.000 palabras) se divide en chunks de
**200 palabras** con un **overlap de 50 palabras** entre chunks consecutivos.

El overlap evita perder información cuando una idea cruza el límite entre dos chunks.

**Total PDF: 109 chunks**

### Total general: 42 + 109 = **151 chunks**

### Flujo de ingesta

```
1. Recorrer data/projects/*/
   a. Leer project_info.yaml → 1 chunk
   b. Leer configs/*.yaml → 4 chunks por modelo

2. Recorrer data/*.pdf
   a. Extraer texto página por página
   b. Dividir en chunks de 200 palabras (overlap 50)

3. Generar embeddings de todos los chunks via Ollama
4. Guardar textos + embeddings + metadatos en ChromaDB

Nota: Si chroma_db/ ya existe con datos, la ingesta se salta automáticamente.
      Usar --reingest para forzar re-procesamiento.
```

---

## El Proceso de Consulta (app.py)

### Flujo de consulta

```
1. Usuario escribe pregunta (con /reference opcional)
2. Se detecta si incluye el comando /reference
3. Se genera embedding de la pregunta (Ollama)
4. ChromaDB busca los 5 chunks más similares (TOP_K=5)
5. Se arma un prompt con formato:
   "Contexto: [chunks encontrados]
    Pregunta: [pregunta del usuario]
    Respuesta:"
6. Se envía al LLM con historial de últimas 4 interacciones
7. Se muestra la respuesta
8. Si se usó /reference, se añaden las fuentes al final
```

### Comandos disponibles en el chat

| Comando | Efecto |
|---------|--------|
| Pregunta normal | Responde sin mostrar fuentes |
| Pregunta + `/reference` | Responde y muestra de qué chunks/archivos sacó la info |

**Ejemplo:**
```
¿Qué es STAMM?                     → Responde sin referencias
¿Qué es STAMM? /reference          → Responde + muestra fuentes con score
```

### Memoria de conversación

El bot recuerda las últimas **4 interacciones** de la conversación.
La búsqueda de chunks siempre usa la pregunta actual (sin historial) para
evitar que conversaciones previas contaminen el retrieval.

### El System Prompt

Le decimos al LLM:
- Que es experto en el registro de modelos y en el paper STAMM
- Que SOLO use la información del contexto (no invente)
- Que responda conciso y en el idioma de la pregunta

---

## Estructura de Archivos

```
Proyecto Final NLP/
└── model-registry-rag/
    ├── app.py               ← RAG completo: ingesta + chatbot + Gradio/Terminal
    ├── requirements.txt     ← Dependencias Python
    ├── chroma_db/           ← Base de datos de vectores (generada automáticamente)
    └── data/
        ├── stamm_paper.pdf  ← Paper STAMM (fuente de conocimiento)
        └── projects/
            ├── Project_IndPenSim/
            │   ├── project_info.yaml
            │   └── configs/ ← 10 YAMLs de modelos
            └── Bioindustry_E.Coli/
                └── project_info.yaml
```

---

## Modos de Ejecución

```bash
# Gradio con Qwen 3.5 9B (UI web, más potente)
python app.py

# Terminal con Qwen 3.5 4B (ligero, sin interfaz)
python app.py --terminal

# Forzar re-ingesta (cuando se añaden nuevos documentos)
python app.py --reingest
```

---

## Decisiones de Diseño Resumidas

| Decisión | Elegido | Razón |
|----------|---------|-------|
| Fuentes de datos | YAMLs + PDF | Metadatos estructurados + paper técnico de referencia |
| Embeddings | nomic-embed-text-v2-moe | Ya instalado en Ollama, 100% local |
| Vector store | ChromaDB | Simple, embedded, suficiente para 151 chunks |
| LLM principal | Qwen 3.5 9B | Potente, corre local via Ollama |
| LLM ligero | Qwen 3.5 4B | Para PCs menos potentes, modo terminal |
| Chunking YAMLs | 4 chunks por modelo (semántico) | Mejora precisión vs. 1 chunk gigante |
| Chunking PDF | 200 palabras + overlap 50 | Estándar para texto no estructurado |
| Thinking mode | Desactivado | 3-5x más rápido sin perder calidad notable |
| Referencias | Comando /reference | Opcionales — no contaminan la UI por defecto |
| Framework | Python puro | Sin LangChain = menos dependencias, más control |
| Costo | $0 | Todo local, sin APIs externas |
