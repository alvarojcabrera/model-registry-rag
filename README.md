# Mini RAG - Model Registry: Guia Paso a Paso

## Requisitos Previos

Antes de clonar y correr este proyecto, necesitas tener instalado:

### 1. Ollama
Descarga e instala desde [ollama.com](https://ollama.com/).
Luego descarga los modelos necesarios:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text-v2-moe
```

> **Nota:** `qwen3.5:9b` requiere ~6-7GB de VRAM. Si no tienes GPU, correra en CPU pero sera mas lento.

### 2. Python 3.10+
Asegurate de tener Python instalado. Puedes verificarlo con:
```bash
python --version
```

### 3. Los archivos YAML del Model Registry
Este RAG lee los metadatos de modelos desde archivos YAML. 

---

## Instalacion

```bash
# 1. Clonar este repositorio
git clone https://github.com/alvarojcabrera/model-registry-rag.git
cd model-registry-rag

# 2. Instalar dependencias de Python
pip install -r requirements.txt

# 3. Asegurate de que Ollama este corriendo
ollama serve

# 4. Ejecutar la ingesta (genera la base de datos de vectores)
python ingest.py

# 5. Opcion A: Chat interactivo por terminal
python query.py

# 5. Opcion B: Interfaz web con Streamlit
streamlit run app.py
```

> **Importante:** El paso 4 (`ingest.py`) requiere que la carpeta `projects/` con los
> YAMLs exista en la ruta esperada. Por defecto busca en `../projects/` (un nivel arriba
> de la carpeta `rag/`). Si clonaste los repos por separado, ajusta la ruta en `ingest.py`.

---

## Que es un RAG?

RAG = **Retrieval-Augmented Generation** (Generacion Aumentada por Recuperacion).

Es una tecnica que combina dos cosas:
1. **Buscar** informacion relevante en tus documentos
2. **Generar** una respuesta usando un LLM, pero alimentandolo con esa informacion encontrada

Sin RAG, el LLM solo sabe lo que aprendio en su entrenamiento.
Con RAG, el LLM puede "leer" TUS datos y responder sobre ellos.

```
┌──────────────────────────────────────────────────────┐
│                  FLUJO DE UNA PREGUNTA                │
│                                                       │
│  Usuario: "Que modelos usan Random Forest?"           │
│       │                                               │
│       ▼                                               │
│  [1] Convertir pregunta → vector (embedding)          │
│       │                                               │
│       ▼                                               │
│  [2] Buscar vectores similares en ChromaDB            │
│       │                                               │
│       ▼                                               │
│  [3] Recuperar los 5 chunks mas relevantes            │
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

**Que son?** Los archivos YAML del model-registry contienen metadatos de cada modelo:
- Nombre, autor, version, fecha
- Tipo de algoritmo (RF, SVM, LSTM, etc.)
- Inputs (features) con rangos esperados
- Outputs (que predice)
- Hiperparametros de entrenamiento

**Por que YAMLs y no PDFs?**
- Ya teniamos datos estructurados y limpios
- No necesitamos OCR ni parseo complejo
- El caso de uso es concreto: consultar metadatos de modelos

### Componente 2: Embeddings (nomic-embed-text-v2-moe)

**Que es un embedding?**
Es convertir texto en un vector numerico (lista de numeros). Textos con significado
similar producen vectores cercanos entre si.

```
"Random Forest con 5 arboles" → [0.12, -0.45, 0.78, ...]
"RF usando 5 estimadores"     → [0.11, -0.44, 0.77, ...]  ← Muy similar!
"Temperatura del reactor"     → [0.89, 0.23, -0.56, ...]  ← Diferente
```

**Por que nomic-embed-text-v2-moe?**
- Ya lo tenia instalado en Ollama
- Corre 100% local y gratis
- Buen rendimiento para textos tecnicos
- Evita instalar PyTorch (~2GB) que necesitaria sentence-transformers

**Alternativa considerada:** `all-MiniLM-L6-v2` (sentence-transformers)
- Requiere instalar torch + sentence-transformers (~3GB extra)

### Componente 3: Vector Store (ChromaDB)

**Que es?**
Una base de datos especializada en buscar por similitud de vectores.
Cuando haces una pregunta, ChromaDB encuentra los chunks cuyos vectores
son mas parecidos al vector de tu pregunta.

**Por que ChromaDB?**
- Instalacion simple: `pip install chromadb` y listo
- No requiere servidor externo (es embedded, corre dentro de Python)
- Persiste en disco (carpeta `chroma_db/`)
- API sencilla: add(), query(), y poco mas

**Alternativas consideradas:**
- FAISS (Facebook): mas rapido para millones de vectores, pero mas complejo de usar
- Pinecone/Weaviate: requieren servidor o son servicios cloud (no gratis)
- Para 42 chunks, ChromaDB es mas que suficiente

### Componente 4: LLM (Qwen 3.5 9B via Ollama)

**Que hace?**
Recibe el contexto recuperado + la pregunta del usuario y genera una respuesta
en lenguaje natural.

**Por que Ollama?**
- Simplifica correr LLMs locales (descarga, cuantizacion, servidor API)
- API REST estandar compatible con cualquier cliente
- Gratis y sin telemetria

---

## El Proceso de Ingesta (ingest.py)

### Que es un "chunk"?

En vez de meter un YAML entero como un solo documento, lo dividimos en pedazos
tematicos (chunks). Esto mejora la precision del retrieval.

Cada modelo genera **4 chunks**:

| Chunk | Contenido | Por que separarlo |
|-------|-----------|-------------------|
| `overview` | Nombre, autor, tipo, lenguaje, descripcion | Preguntas generales sobre el modelo |
| `training` | Instancias, hiperparametros, validacion | Preguntas sobre entrenamiento |
| `inputs` | Features con tipos, unidades, rangos | Preguntas sobre que datos necesita |
| `outputs` | Variables de salida, rangos | Preguntas sobre que predice |

Cada `project_info.yaml` genera **1 chunk** con info del proyecto.

**Total: 10 modelos x 4 chunks + 2 proyectos = 42 chunks**

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
3. ChromaDB busca los 5 chunks mas similares (TOP_K=5)
4. Se arma un prompt con formato:
   "Contexto: [chunks encontrados]
    Pregunta: [pregunta del usuario]
    Respuesta:"
5. Se envia al LLM (Qwen 3.5) via Ollama
6. Se muestra la respuesta en streaming (token por token)
```

### El System Prompt

Le decimos al LLM:
- Que es experto en el registro de modelos
- Que SOLO use la informacion del contexto (no invente)
- Que responda conciso y en el idioma de la pregunta

---

## Estructura de Archivos

```
model-registry-rag/
├── ingest.py           ← Parsea YAMLs y los mete en ChromaDB
├── query.py            ← Chat interactivo por terminal
├── app.py              ← Interfaz web con Streamlit
├── requirements.txt    ← Dependencias Python
├── GUIA_RAG.md         ← Guia detallada del proyecto
├── README.md           ← Este archivo
├── .gitignore          ← Excluye chroma_db/, __pycache__, etc.
└── chroma_db/          ← Base de datos de vectores (generada, no se sube)
```

> **Nota:** La carpeta `chroma_db/` no se incluye en el repositorio. Se genera
> automaticamente al ejecutar `python ingest.py`.

---

## Decisiones de Diseno Resumidas

| Decision | Elegido | Razon |
|----------|---------|-------|
| Fuente de datos | YAMLs del registry | Datos limpios y estructurados ya existentes |
| Embeddings | nomic-embed-text-v2-moe | Ya instalado en Ollama, 100% local |
| Vector store | ChromaDB | Simple, embedded, suficiente para ~42 chunks |
| LLM | Qwen 3.5 9B | Elegido por el usuario, corre local via Ollama |
| Chunking | 4 chunks por modelo | Mejora precision vs. 1 chunk gigante por modelo |
| Framework | Python puro | Sin LangChain = menos dependencias, mas control |
| Costo | $0 | Todo local, sin APIs externas |

---

## Stack Tecnologico

| Componente | Herramienta | Costo |
|---|---|---|
| LLM | Qwen 3.5 9B via Ollama | Gratis, local |
| Embeddings | nomic-embed-text-v2-moe via Ollama | Gratis, local |
| Vector Store | ChromaDB | Gratis, local |
| Interfaz Web | Streamlit | Gratis, local |
| Lenguaje | Python | Gratis |

**Costo total: $0** - 
