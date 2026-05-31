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
│  1. Usuario escribe pregunta                                  │
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
│  6. Respuesta al usuario (+ /reference si se pidió)           │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes del Sistema

### 2.1 Fuentes de Datos (Base de Conocimiento)

| Archivo | Tipo | Chunks | Método de chunking |
|---------|------|--------|-------------------|
| `0001_[python]_penicillin_RF.yaml` ... (10 configs) | YAML | 40 | Semántico: 4 secciones por modelo |
| `project_info.yaml` (2 proyectos) | YAML | 2 | 1 chunk por proyecto |
| `stamm_paper.pdf` | PDF | 109 | Por palabras: 200 words, overlap 50 |
| `Manual_Evaluacion_RAG_IndPenSim_PRO.docx` | DOCX | Variable | Por párrafos |
| `Rubrica de calificación de los tres modelos.xlsx` | XLSX | Variable | Por pregunta/respuesta |
| **Total** | | **~200+** | |

### 2.2 Embeddings

**Modelo:** `nomic-embed-text-v2-moe` (via Ollama)

Convierte texto en vectores numéricos de alta dimensión. Textos semánticamente similares producen vectores cercanos en el espacio vectorial.

```
"Random Forest con 100 árboles"  → [0.12, -0.45, 0.78, ...]
"RF usando n_estimators=100"     → [0.11, -0.44, 0.77, ...]  ← Muy similar
"Temperatura del bioreactor"     → [0.89, 0.23, -0.56, ...]  ← Diferente
```

**¿Por qué este modelo?**
- Ya disponible en Ollama (sin descargas extra)
- 100% local — sin APIs externas
- Buen rendimiento para texto técnico/científico

### 2.3 Vector Store — ChromaDB

Base de datos especializada en búsqueda vectorial. Almacena:
- El texto de cada chunk
- Su embedding (vector)
- Metadatos (fuente, tipo de documento)

Usa índice **HNSW** (Hierarchical Navigable Small World) con similitud coseno para búsquedas eficientes.

### 2.4 LLMs disponibles

| Modo | Modelo | Parámetros | Interfaz | Comando |
|------|--------|-----------|----------|---------|
| Completo | Qwen 3.5 9B | 9B | Gradio (web) | `python app.py` |
| Ligero | Qwen 3.5 4B | 4B | Terminal | `python app.py --terminal` |

**Optimizaciones aplicadas:**
- `think: False` → Desactiva razonamiento interno de Qwen (3-5x más rápido)
- `num_predict: 300` → Limita longitud de respuesta
- Historial limitado a últimas 4 interacciones

---

## 3. ¿Qué es un LLM?

**LLM = Large Language Model** (Modelo de Lenguaje Grande).

Es una red neuronal entrenada sobre enormes cantidades de texto que aprende a predecir el siguiente token (palabra/subpalabra) dado un contexto. A través de este entrenamiento emergente desarrolla capacidades de razonamiento, síntesis y generación de texto.

### Arquitectura base: Transformer

Todos los LLMs modernos se basan en la arquitectura **Transformer** (2017, Vaswani et al.):
- **Atención multi-cabeza**: permite al modelo relacionar palabras distantes en el texto
- **Encoder/Decoder**: procesa entrada y genera salida
- **Self-attention**: cada token atiende a todos los demás tokens del contexto

### Parámetros y escala

| Modelo | Parámetros | VRAM aprox. | Característica |
|--------|-----------|------------|----------------|
| Qwen 3.5 4B | 4B | ~3 GB | Rápido, ligero |
| Qwen 3.5 9B | 9B | ~6.6 GB | Mejor calidad |
| Llama 3 8B | 8B | ~5 GB | Usado en Colab |
| GPT-5.5 | No public | API cloud | OpenAI |
| Gemini 2.5 Flash | No public | API cloud | Google |
| Meta AI | No public | API cloud | Meta |

### Cuantización

Los modelos se comprimen (cuantizan) para caber en menos VRAM. Ollama maneja esto automáticamente. Qwen 3.5 9B de 6.6GB es la versión Q4 (4 bits), que ocupa ~4x menos que la versión completa en float32.

---

## 4. Evaluación Comparativa de LLMs

### 4.1 Metodología

Se evaluaron **3 LLMs externos** usando el mismo sistema RAG con las mismas preguntas:

| LLM | Proveedor | Acceso |
|-----|-----------|--------|
| Gemini 2.5 Flash | Google | Online |
| Meta AI | Meta | Online |
| GPT-5.5 Instant | OpenAI | Online |

### 4.2 Tipos de preguntas

**Preguntas trampa (10)** — verifican que el modelo NO alucine:
- "¿Qué modelo usa imágenes?"
- "¿Qué modelo usa reconocimiento de voz?"
- "¿Qué modelo tiene accuracy del 99%?"
- "¿Qué modelo fue entrenado con datos de satélite?"

Un buen RAG debe responder "No se encontró esa información en el contexto" en lugar de inventar.

**Preguntas normales (5)** — verifican recuperación correcta:
- "¿Qué diferencias existen entre CART, Random Forest y GBM?"
- "¿Qué modelos usan Python vs R?"
- "¿Qué modelo es más interpretable?"
- "¿Qué modelo recomendarías en presencia de ruido?"
- "¿Qué implicaciones tiene usar LSTM?"

### 4.3 Rúbrica de calificación

Cada respuesta se evalúa en 6 criterios (escala 1-5 cada uno, máximo 30 por pregunta):

| Criterio | Descripción |
|----------|-------------|
| **Precisión** | La respuesta es factualmente correcta |
| **Uso del contexto** | Usa la información del RAG correctamente |
| **Coherencia** | La respuesta es lógica y bien estructurada |
| **Interpretación** | Interpreta correctamente la pregunta |
| **No alucinación** | No inventa información que no existe |
| **Claridad** | La respuesta es fácil de entender |

**Escala:** 1 = Muy deficiente | 3 = Aceptable | 5 = Excelente

### 4.4 Resultados

**15 preguntas × 30 puntos = 450 puntos máximo por modelo**

| Modelo | Puntaje | Máximo | Porcentaje |
|--------|---------|--------|-----------|
| Google (Gemini 2.5 Flash) | 436 | 450 | **96.9%** |
| Meta AI online | 450 | 450 | **100.0%** |
| OpenAI (GPT-5.5 Instant) | 450 | 450 | **100.0%** |

**Desglose por tipo:**

| Tipo | Gemini | Meta AI | GPT-5.5 |
|------|--------|---------|---------|
| Preguntas trampa (10×30=300) | 290/300 (96.7%) | 300/300 (100%) | 300/300 (100%) |
| Preguntas normales (5×30=150) | 146/150 (97.3%) | 150/150 (100%) | 150/150 (100%) |

**Observación:** Gemini perdió 1 punto en varias preguntas en el criterio de "Interpretación" (calificado 4/5), mientras que Meta AI y GPT-5.5 obtuvieron puntaje perfecto.

---

## 5. Decisiones de Diseño del RAG

| Decisión | Elegido | Alternativa descartada | Razón |
|----------|---------|----------------------|-------|
| Embeddings | nomic-embed-text-v2-moe (Ollama) | sentence-transformers (PyTorch) | Sin dependencias pesadas, 100% local |
| Vector Store | ChromaDB | FAISS / Pinecone | Simple, embedded, sin servidor |
| LLM local | Qwen 3.5 9B/4B | Llama 3 / Mistral | Mejor rendimiento en español/inglés técnico |
| Chunking YAMLs | Semántico por secciones | Por palabras fijas | Preserva coherencia temática |
| Chunking PDF | 200 palabras + overlap 50 | Sin overlap | Evita perder información en bordes |
| Thinking mode | Desactivado | Activado | 3-5x más rápido sin pérdida notable de calidad |
| Framework | Python puro | LangChain | Menos dependencias, más control y transparencia |
| Costo | $0 | APIs cloud ($$$) | Todo local, sin cobros externos |

---

## 6. Cómo correr el sistema

### Prerequisitos
```bash
# Ollama debe estar instalado y corriendo
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull nomic-embed-text-v2-moe
```

### Comandos
```bash
cd "D:\Escritorio\Proyecto Final NLP\model-registry-rag"

# Modo completo: Gradio + Qwen 9B
& "C:\Users\alvar\AppData\Local\Programs\Python\Python311\python.exe" app.py

# Modo ligero: Terminal + Qwen 4B
& "C:\Users\alvar\AppData\Local\Programs\Python\Python311\python.exe" app.py --terminal

# Forzar re-ingesta (cuando se añaden nuevos documentos)
& "C:\Users\alvar\AppData\Local\Programs\Python\Python311\python.exe" app.py --reingest
```

### Comandos en el chat
| Comando | Efecto |
|---------|--------|
| Pregunta normal | Responde sin mostrar fuentes |
| Pregunta + `/reference` | Responde mostrando fuentes y scores de similitud |
