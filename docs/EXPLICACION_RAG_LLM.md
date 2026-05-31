# Explicación del Sistema RAG y los LLMs Evaluados

---

## 1. ¿Qué es un RAG?

**RAG = Retrieval-Augmented Generation** (Generación Aumentada por Recuperación).

Es una técnica que resuelve una limitación fundamental de los LLMs: **no conocen tus datos privados ni información actualizada**. La solución es darle al modelo el contexto relevante justo en el momento de la pregunta.

```
Sin RAG:
  Usuario: "¿Qué hiperparámetros usa el modelo 0001?"
  LLM:     "No tengo esa información." ← Solo conoce lo del entrenamiento

Con RAG:
  Usuario: "¿Qué hiperparámetros usa el modelo 0001?"
  Sistema: [busca en ChromaDB] → encuentra chunk con los hiperparámetros
  LLM:     "El modelo 0001 usa n_estimators=100, max_depth=5..." ← Responde con TUS datos
```

### Flujo completo de una pregunta

```
┌──────────────────────────────────────────────────────────────┐
│                    PIPELINE RAG COMPLETO                      │
│                                                               │
│  1. Usuario escribe pregunta (+ /reference opcional)          │
│       │                                                       │
│       ▼                                                       │
│  2. Embedding de la pregunta                                  │
│     (nomic-embed-text-v2-moe via Ollama)                      │
│       │                                                       │
│       ▼                                                       │
│  3. Búsqueda por similitud coseno en ChromaDB                 │
│     → Devuelve TOP-K=5 chunks más relevantes                  │
│       │                                                       │
│       ▼                                                       │
│  4. Construcción del prompt:                                  │
│     [System Prompt] + [Historial últimas 4] + [Contexto] + Q │
│       │                                                       │
│       ▼                                                       │
│  5. LLM genera respuesta (Qwen 3.5 9B o 4B via Ollama)       │
│       │                                                       │
│       ▼                                                       │
│  6. Respuesta al usuario (+ fuentes si se usó /reference)     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes del Sistema

### 2.1 Fuentes de Datos (Base de Conocimiento)

El sistema ingesta **4 tipos de documentos** automáticamente desde la carpeta `data/`:

| Archivo | Tipo | Chunks generados | Método de chunking |
|---------|------|-----------------|-------------------|
| 10 configs de modelos ML | YAML | 40 | Semántico: 4 secciones por modelo |
| 2 `project_info.yaml` | YAML | 2 | 1 chunk por proyecto |
| `stamm_paper.pdf` | PDF | 109 | Por palabras: 200 words, overlap 50 |
| `Manual_Evaluacion_RAG_IndPenSim_PRO.docx` | DOCX | 3 | Por palabras: 150 words, overlap 30 |
| `Rubrica de calificación de los tres modelos.xlsx` | XLSX | 17 | Por pregunta/respuesta/resumen |
| **Total real** | | **171 chunks** | |

### 2.2 Estrategia de Chunking

**¿Por qué diferentes estrategias según el tipo de archivo?**

- **YAMLs → Chunking semántico por secciones**: Los YAMLs tienen estructura definida (identificación, entrenamiento, inputs, outputs). Dividirlos por sección garantiza que cada chunk tenga coherencia temática completa. Un chunk de "inputs" solo habla de inputs, nunca se mezcla con hiperparámetros.

- **PDF y DOCX → Chunking por palabras con overlap**: El texto no estructurado se parte en ventanas de palabras fijas. El overlap (solapamiento) entre chunks evita perder información cuando una idea cruza el límite entre dos chunks.

- **XLSX → Chunking por fila/pregunta**: Cada fila del Excel (pregunta + respuestas de los 3 LLMs) se convierte en un chunk independiente, más chunks adicionales para el resumen de puntajes y la rúbrica.

### 2.3 Embeddings

**Modelo:** `nomic-embed-text-v2-moe` (via Ollama)

Convierte texto en vectores numéricos de alta dimensión. Textos semánticamente similares producen vectores cercanos en el espacio vectorial.

```
"Random Forest con 100 árboles"  → [0.12, -0.45, 0.78, ...]
"RF usando n_estimators=100"     → [0.11, -0.44, 0.77, ...]  ← Muy similar
"Temperatura del bioreactor"     → [0.89, 0.23, -0.56, ...]  ← Diferente
```

**¿Por qué este modelo y no sentence-transformers?**
- Ya disponible en Ollama — sin descargas extra ni instalación de PyTorch (~2GB)
- 100% local — sin APIs externas ni costos
- Buen rendimiento para texto técnico/científico en español e inglés

### 2.4 Vector Store — ChromaDB

Base de datos especializada en búsqueda vectorial. Almacena:
- El texto de cada chunk
- Su embedding (vector numérico)
- Metadatos: archivo fuente, tipo de documento (`model_overview`, `pdf_paper`, `xlsx_evaluacion`, etc.)

Usa índice **HNSW** (Hierarchical Navigable Small World) con similitud coseno para búsquedas eficientes. La DB persiste en disco en `chroma_db/` — no se regenera en cada arranque a menos que se use `--reingest`.

**¿Por qué ChromaDB y no FAISS o Pinecone?**
- ChromaDB: instalación con `pip install chromadb`, sin servidor externo, embedded dentro de Python
- FAISS: más rápido para millones de vectores pero más complejo de usar
- Pinecone/Weaviate: cloud, requieren registro y tienen costos

Para 171 chunks, ChromaDB es más que suficiente.

### 2.5 LLMs disponibles

| Modo | Modelo | Parámetros | VRAM aprox. | Interfaz | Comando |
|------|--------|-----------|------------|----------|---------|
| Completo | Qwen 3.5 9B | 9B | ~6.6 GB | Gradio (UI web) | `python app.py` |
| Ligero | Qwen 3.5 4B | 4B | ~2.5 GB | Terminal | `python app.py --terminal` |

**Optimizaciones aplicadas:**
- `think: False` → Desactiva el razonamiento interno de Qwen 3.5 (modo "thinking"). Reduce el tiempo de respuesta 3-5x sin pérdida notable de calidad.
- `num_predict: 300` → Limita la respuesta a 300 tokens máximo.
- Historial limitado a las últimas **4 interacciones** del chat.

---

## 3. ¿Qué es un LLM?

**LLM = Large Language Model** (Modelo de Lenguaje Grande).

Es una red neuronal entrenada sobre enormes cantidades de texto que aprende a predecir el siguiente token dado un contexto. A través de este entrenamiento emergen capacidades de razonamiento, síntesis y generación de texto.

### Arquitectura base: Transformer

Todos los LLMs modernos se basan en la arquitectura **Transformer** (Vaswani et al., 2017):
- **Atención multi-cabeza**: permite al modelo relacionar palabras distantes en el texto
- **Self-attention**: cada token atiende a todos los demás tokens del contexto
- **Feed-forward layers**: transformaciones no lineales que codifican conocimiento

### Parámetros y escala

| Modelo | Parámetros | Acceso | Característica |
|--------|-----------|--------|----------------|
| Qwen 3.5 4B | 4B | Local (Ollama) | Rápido, ligero — modo terminal |
| Qwen 3.5 9B | 9B | Local (Ollama) | Mayor calidad — modo Gradio |
| Llama 3 8B | 8B | Local (Ollama) | Usado en versión Colab |
| Gemini 2.5 Flash | No público | API Google | Evaluado en la rúbrica |
| Meta AI | No público | API Meta | Evaluado en la rúbrica |
| GPT-5.5 Instant | No público | API OpenAI | Evaluado en la rúbrica |

### Cuantización

Los modelos se comprimen (cuantizan) para caber en menos VRAM. Ollama maneja esto automáticamente. Qwen 3.5 9B a 6.6 GB es la versión Q4 (4 bits por peso), que ocupa ~4x menos que la versión completa en float32 sin pérdida de calidad significativa.

---

## 4. Evaluación Comparativa de LLMs

### 4.1 Objetivo

Verificar qué tan bien responden diferentes LLMs (externos y de pago) cuando se les da el mismo contexto que el RAG recupera. Esto permite comparar objetivamente la calidad de respuesta entre modelos bajo las mismas condiciones.

### 4.2 Modelos evaluados

| LLM | Proveedor | Tipo de acceso |
|-----|-----------|---------------|
| Gemini 2.5 Flash | Google | API cloud (online) |
| Meta AI | Meta | API cloud (online) |
| GPT-5.5 Instant | OpenAI | API cloud (online) |

> Estos modelos **no corren localmente** — son servicios externos. El RAG local usa Qwen 3.5 vía Ollama.

### 4.3 Tipos de preguntas usadas

**Preguntas trampa (10)** — verifican que el modelo NO alucine información inexistente:

| # | Pregunta |
|---|---------|
| 1 | ¿Qué modelo tiene accuracy del 99%? |
| 2 | ¿Qué modelo usa imágenes? |
| 3 | ¿Qué modelo usa redes convolucionales? |
| 4 | ¿Qué modelo fue entrenado con datos de audio? |
| 5 | ¿Qué modelo usa visión por computador para monitorear la fermentación? |
| 6 | ¿Qué modelo clasifica bacterias en categorías? |
| 7 | ¿Qué modelo tiene precisión perfecta en todos los batches? |
| 8 | ¿Qué modelo predice la penicilina usando video en tiempo real? |
| 9 | ¿Qué modelo fue entrenado con datos de satélite para la fermentación? |
| 10 | ¿Qué modelo usa reconocimiento de voz para alertar al operario? |

Un buen RAG responde: *"No se encontró esa información en el contexto"* — no inventa.

**Preguntas normales (5)** — verifican recuperación y razonamiento correctos:

| # | Pregunta |
|---|---------|
| 11 | ¿Qué diferencias existen entre CART, Random Forest y GBM? |
| 12 | ¿Qué modelos usan Python vs R? |
| 13 | ¿Qué modelo es más interpretable? |
| 14 | ¿Qué modelo recomendarías en presencia de ruido? |
| 15 | ¿Qué implicaciones tiene usar LSTM? |

### 4.4 Rúbrica de calificación

Cada respuesta se evalúa en 6 criterios (escala 1-5 cada uno = máximo **30 puntos** por pregunta):

| Criterio | Descripción |
|----------|-------------|
| **Precisión** | La respuesta es factualmente correcta |
| **Uso del contexto** | Utiliza correctamente la información del RAG |
| **Coherencia** | La respuesta es lógica y bien estructurada |
| **Interpretación** | Interpreta correctamente la intención de la pregunta |
| **No alucinación** | No inventa información que no existe en el contexto |
| **Claridad** | La respuesta es fácil de entender |

Escala: **1** = Muy deficiente | **3** = Aceptable | **5** = Excelente

### 4.5 Resultados reales

**15 preguntas × 30 puntos = 450 puntos máximo por modelo**

| Modelo | Puntaje total | Máximo | Porcentaje |
|--------|-------------|--------|-----------|
| Google (Gemini 2.5 Flash) | 436 | 450 | **96.9%** |
| Meta AI online | 450 | 450 | **100.0%** |
| OpenAI (GPT-5.5 Instant) | 450 | 450 | **100.0%** |

**Desglose por tipo de pregunta:**

| Tipo | Gemini | Meta AI | GPT-5.5 |
|------|--------|---------|---------|
| Preguntas trampa (10 × 30 = 300 pts) | 290/300 (96.7%) | 300/300 (100%) | 300/300 (100%) |
| Preguntas normales (5 × 30 = 150 pts) | 146/150 (97.3%) | 150/150 (100%) | 150/150 (100%) |

**Análisis:** Gemini perdió 14 puntos en el criterio de **Interpretación** (calificado 4/5 en varias preguntas), mientras que Meta AI y GPT-5.5 obtuvieron puntaje perfecto. Los tres modelos respondieron correctamente las preguntas trampa sin alucinar.

---

## 5. Decisiones de Diseño

| Decisión | Elegido | Alternativa descartada | Razón |
|----------|---------|----------------------|-------|
| Embeddings | nomic-embed-text-v2-moe (Ollama) | sentence-transformers (PyTorch) | Sin dependencias pesadas, 100% local |
| Vector Store | ChromaDB | FAISS / Pinecone | Simple, embedded, sin servidor externo |
| LLM local principal | Qwen 3.5 9B | Llama 3 / Mistral | Mejor rendimiento en español/inglés técnico |
| LLM local ligero | Qwen 3.5 4B | Llama 3 2B | Balance velocidad/calidad para terminal |
| Chunking YAMLs | Semántico por secciones | Por palabras fijas | Preserva coherencia temática de cada sección |
| Chunking PDF/DOCX | 200/150 palabras + overlap | Sin overlap | Evita perder información en bordes de chunks |
| Chunking XLSX | Por pregunta/respuesta | Por palabras | Preserva la estructura de evaluación |
| Thinking mode | Desactivado (`think: False`) | Activado | 3-5x más rápido sin pérdida notable de calidad |
| Referencias | Comando `/reference` | Siempre visibles | No contaminan la UI por defecto |
| Memoria | Últimas 4 interacciones en prompt | Reformulación con historial | Evita contaminación del retrieval |
| Framework | Python puro | LangChain | Menos dependencias, más control y transparencia |
| Costo total | $0 | APIs cloud (pago por token) | Todo local, sin cobros externos |

---

## 6. Comandos del sistema

```bash
# Gradio + Qwen 9B (recomendado)
python app.py

# Terminal + Qwen 4B (ligero)
python app.py --terminal

# Forzar re-ingesta de todos los documentos
python app.py --reingest
```

### Comandos especiales en el chat

| Comando | Resultado |
|---------|-----------|
| Pregunta normal | Solo la respuesta |
| Pregunta + `/reference` | Respuesta + fuentes con score de similitud coseno |
