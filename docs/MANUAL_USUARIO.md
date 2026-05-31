# Manual de Usuario — Mini RAG Model Registry

---

## ¿Qué hace esta aplicación?

Es un **asistente de chat inteligente** que responde preguntas en lenguaje natural sobre:

- Los **modelos de Machine Learning** registrados para el proyecto de producción de penicilina (IndPenSim)
- El **paper STAMM** (framework de monitoreo de modelos para bioprocesos)
- La **evaluación comparativa** de LLMs (Gemini, Meta AI, GPT-5.5)
- La **rúbrica** usada para calificar las respuestas de esos LLMs

Todo corre **en tu PC, sin internet y sin costo**.

---

## Instalación

### Requisitos previos

Antes de instalar, necesitas:

1. **Python 3.11** → [descargar aquí](https://www.python.org/downloads/)
2. **Ollama** → [descargar aquí](https://ollama.com/)

### Paso 1 — Clonar el repositorio

Abre una terminal y ejecuta:

```bash
git clone https://github.com/alvarojcabrera/model-registry-rag.git
cd model-registry-rag
```

### Paso 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3 — Descargar los modelos de lenguaje

```bash
# Modelo principal (6.6 GB — necesita buena GPU)
ollama pull qwen3.5:9b

# Modelo ligero opcional (2.5 GB — para PCs menos potentes)
ollama pull qwen3.5:4b

# Modelo de embeddings (imprescindible)
ollama pull nomic-embed-text-v2-moe
```

> ⚠️ La descarga puede tardar varios minutos según tu conexión.

---

## Cómo ejecutar la aplicación

Abre una terminal, navega a la carpeta del proyecto y ejecuta:

### Modo completo (recomendado)

```bash
# En Mac/Linux
python app.py

# En Windows (PowerShell)
& "C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python311\python.exe" app.py
```

Abre tu navegador en **http://127.0.0.1:7860**

### Modo ligero (para PCs con poca VRAM)

```bash
# En Mac/Linux
python app.py --terminal

# En Windows (PowerShell)
& "C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python311\python.exe" app.py --terminal
```

El chat aparece directamente en la terminal.

---

## Primera vez que ejecutas

La **primera vez** que corres la app, verás esto en la terminal — es normal y solo ocurre una vez:

```
[1/3] Ingesta de YAMLs + PDFs...
  Procesando YAMLs...
  Procesando PDFs...
    stamm_paper.pdf: 109 chunks
  Procesando DOCXs...
    Manual_Evaluacion_RAG_IndPenSim_PRO.docx: 3 chunks
  Procesando XLSXs...
    Rubrica de calificación de los tres modelos.xlsx: 17 chunks
  171 chunks totales
  Generando embeddings...
```

Esto puede tardar **2-5 minutos**. Las siguientes veces arranca en segundos.

---

## Interfaz de chat (Gradio)

Al abrir `http://127.0.0.1:7860` verás:

```
┌─────────────────────────────────────────────┐
│  Mini RAG - Model Registry + STAMM Paper    │
│                                             │
│  [Área de conversación]                     │
│                                             │
│  [Tu pregunta aquí...]          [Enviar]    │
│                                             │
│  [Limpiar conversación]                     │
└─────────────────────────────────────────────┘
```

- **Escribe tu pregunta** y presiona Enter o el botón Enviar
- **Limpiar conversación** borra el historial y reinicia el contexto

---

## Cómo hacer preguntas

### Preguntas sobre los modelos registrados

```
¿Qué modelos hay registrados?
¿Cuál es la diferencia entre CART y Random Forest?
¿Qué modelos usan Python y cuáles usan R?
¿Cuáles son los inputs del modelo LSTM?
¿Qué hiperparámetros usa el SVM?
¿Qué predice el modelo GBM?
¿Qué modelo tiene mejor desempeño predictivo?
¿Qué modelo recomendarías para datos con ruido?
¿Qué modelos son interpretables y cuáles son caja negra?
```

### Preguntas sobre el paper STAMM

```
¿Qué es STAMM?
¿Quiénes crearon el framework STAMM?
¿Para qué sirve STAMM en la industria?
¿Qué es un soft sensor?
¿Cómo se relaciona STAMM con el model registry?
```

### Preguntas sobre la evaluación de LLMs

```
¿Qué puntaje obtuvo Gemini en la evaluación?
¿Cuáles fueron las preguntas trampa de la evaluación?
¿Qué criterios se usaron para calificar las respuestas?
¿Qué modelo LLM tuvo el mejor desempeño en la rúbrica?
¿Cuántos puntos vale cada pregunta de la evaluación?
```

### Preguntas de seguimiento (el bot recuerda el contexto)

Puedes hacer preguntas encadenadas:

```
Usuario: ¿Qué modelos usan Random Forest?
Bot:     [responde con los modelos RF]

Usuario: ¿En qué lenguaje están implementados?
Bot:     [entiende que hablas de los RF y responde]
```

---

## Comando especial: /reference

Si quieres saber **de dónde viene la información** en la respuesta, agrega `/reference` al final de tu pregunta:

```
¿Qué es STAMM? /reference
```

La respuesta incluirá las fuentes con su nivel de relevancia:

```
STAMM es un framework para...

Referencias:
- stamm_paper.pdf (pdf_paper, score: 0.847)
- project_info.yaml (project_info, score: 0.612)
```

El **score** (0 a 1) indica qué tan relevante fue ese documento para tu pregunta. Más cercano a 1 = más relevante.

---

## Tipos de archivos que conoce el bot

| Fuente | Qué contiene |
|--------|-------------|
| YAMLs de modelos | Nombre, autor, hiperparámetros, inputs, outputs, lenguaje |
| `stamm_paper.pdf` | Paper científico completo sobre el framework STAMM |
| `Manual_Evaluacion_RAG_IndPenSim_PRO.docx` | Preguntas de evaluación del RAG |
| `Rubrica de calificación.xlsx` | Respuestas de Gemini/Meta AI/GPT-5.5 y sus puntajes |

---

## Situaciones comunes

### El bot dice "no encontré esa información"

Correcto — el bot **solo responde con lo que está en los documentos**. Si preguntas algo que no está en los archivos, lo dice claramente en lugar de inventar. Esto es una característica, no un error.

### La respuesta es lenta

- La primera pregunta siempre es más lenta (el modelo carga en memoria)
- Las siguientes preguntas son más rápidas
- Si tu PC no tiene GPU dedicada, las respuestas pueden tardar 30-90 segundos

### Quiero olvidar la conversación anterior

Haz clic en **"Limpiar conversación"** en la interfaz, o escribe una nueva pregunta sin relación con la anterior.

### Cambié documentos en `data/` y quiero que el bot los aprenda

Cierra la app y vuelve a correr con `--reingest`:

```bash
python app.py --reingest
```

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'chromadb'` | Ejecuta `pip install -r requirements.txt` con el Python correcto |
| `Connection refused` al embedder | Verifica que Ollama esté corriendo: `ollama list` |
| `Could not create share link` | Es un problema temporal de Gradio. El link local `http://127.0.0.1:7860` sigue funcionando |
| El kernel crashea en Jupyter/VS Code | Corre `app.py` directamente desde terminal, no desde Jupyter |
| `KeyError: '_type'` en ChromaDB | Borra la carpeta `chroma_db/` y vuelve a correr la app |

---

## Versión para Google Colab

Si no tienes Ollama instalado, puedes usar la versión Colab disponible en:

```
notebooks/Collab.ipynb
```

Ábrelo en Google Colab y sigue las instrucciones de cada celda. La primera ejecución instala Ollama y descarga los modelos automáticamente (~10-15 min).
