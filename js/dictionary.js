const dictionaryData = [
  {
    id: "algoritmo-generativo",
    term: "Algoritmo Generativo",
    shortDefinition: "Sistema computacional que crea contenido nuevo a partir de patrones aprendidos.",
    fullDefinition: "Sistema computacional diseñado para crear contenido nuevo a partir de patrones aprendidos de datos existentes, en lugar de seguir instrucciones explícitas. Estos algoritmos pueden generar texto, imágenes, música, código y otros tipos de contenido.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["ia-generativa", "aprendizaje-profundo"],
    tags: ["core", "generación"]
  },
  {
    id: "aprendizaje-profundo",
    term: "Aprendizaje Profundo (Deep Learning)",
    shortDefinition: "Subconjunto del aprendizaje automático que usa redes neuronales con múltiples capas.",
    fullDefinition: "Subconjunto del aprendizaje automático que utiliza redes neuronales con múltiples capas para analizar diversos factores de datos con una estructura similar a la del cerebro humano.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["red-neuronal", "algoritmo-generativo"],
    tags: ["ml", "redes"]
  },
  {
    id: "arquitectura-transformer",
    term: "Arquitectura Transformer",
    shortDefinition: "Diseño de red neuronal con mecanismos de atención para procesamiento de secuencias.",
    fullDefinition: "Diseño de red neuronal introducido por Google en 2017 que revolucionó el procesamiento del lenguaje natural. Utiliza mecanismos de atención para procesar secuencias de datos en paralelo.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["atencion", "bert", "gpt"],
    tags: ["nlp", "arquitectura"]
  },
  {
    id: "atencion",
    term: "Atención (Attention Mechanism)",
    shortDefinition: "Permite al modelo enfocarse en partes relevantes de la entrada al generar la salida.",
    fullDefinition: "Componente clave de las arquitecturas transformer que permite al modelo enfocarse en diferentes partes de los datos de entrada al generar cada elemento de salida.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["arquitectura-transformer"],
    tags: ["nlp", "mecanismo"]
  },
  {
    id: "autoregresion",
    term: "Autoregresión",
    shortDefinition: "Técnica donde la salida se genera secuencialmente, condicionada por la salida previa.",
    fullDefinition: "Técnica utilizada en modelos generativos donde la salida se genera secuencialmente, con cada nuevo elemento condicionado por los elementos previamente generados.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["gpt", "llm"],
    tags: ["generación", "secuencial"]
  },
  {
    id: "bert",
    term: "BERT (Bidirectional Encoder Representations from Transformers)",
    shortDefinition: "Modelo de lenguaje de Google que procesa texto bidireccionalmente.",
    fullDefinition: "Modelo de lenguaje desarrollado por Google que procesa texto de manera bidireccional, considerando el contexto tanto anterior como posterior de cada palabra.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["arquitectura-transformer", "nlp", "embedding"],
    tags: ["google", "nlp", "modelo"]
  },
  {
    id: "cadena-de-pensamientos",
    term: "Cadena de Pensamientos (Chain of Thought)",
    shortDefinition: "Técnica de prompt engineering que fomenta el razonamiento paso a paso del LLM.",
    fullDefinition: "Técnica de ingeniería de instrucciones que fomenta que un modelo de lenguaje extenso explique su razonamiento paso a paso, mejorando la precisión en tareas complejas.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["prompt-engineering", "llm"],
    tags: ["prompting", "razonamiento"]
  },
  {
    id: "chatgpt",
    term: "ChatGPT",
    shortDefinition: "Modelo conversacional de OpenAI basado en GPT, optimizado para diálogos.",
    fullDefinition: "Modelo conversacional desarrollado por OpenAI basado en la arquitectura GPT, optimizado específicamente para mantener diálogos coherentes y útiles con humanos.",
    category: "aplicaciones",
    complexity: 1,
    relatedTerms: ["gpt", "openai", "llm"],
    tags: ["openai", "chatbot", "conversacional"]
  },
  {
    id: "claude",
    term: "Claude",
    shortDefinition: "Asistente de IA conversacional de Anthropic enfocado en seguridad y utilidad.",
    fullDefinition: "Asistente de IA conversacional desarrollado por Anthropic, diseñado con un enfoque en seguridad, honestidad y utilidad mediante la técnica de \"IA Constitucional\".",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["anthropic", "rlhf", "alineacion-ia"],
    tags: ["anthropic", "chatbot", "seguridad"]
  },
  {
    id: "codificador",
    term: "Codificador (Encoder)",
    shortDefinition: "Componente de red neuronal que transforma entradas en representaciones vectoriales.",
    fullDefinition: "Componente de una arquitectura de red neuronal que transforma datos de entrada en representaciones vectoriales de alta dimensión (embeddings).",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["decodificador", "embedding", "arquitectura-transformer"],
    tags: ["arquitectura", "redes"]
  },
  {
    id: "decodificador",
    term: "Decodificador (Decoder)",
    shortDefinition: "Componente de red neuronal que transforma vectores internos en datos de salida.",
    fullDefinition: "Componente de una arquitectura de red neuronal que transforma representaciones vectoriales internas en datos de salida como texto generado.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["codificador", "arquitectura-transformer", "generacion-condicionada"],
    tags: ["arquitectura", "redes", "generación"]
  },
  {
    id: "difusion",
    term: "Difusión (Diffusion Models)",
    shortDefinition: "Modelos que añaden ruido y aprenden a revertirlo, usados para generar imágenes.",
    fullDefinition: "Clase de modelos generativos que funcionan añadiendo ruido gradualmente a los datos y luego aprendiendo a revertir este proceso, utilizados principalmente para generación de imágenes.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["stable-diffusion", "generacion-condicionada", "imagen-latente"],
    tags: ["imágenes", "generación", "modelo"]
  },
  {
    id: "dall-e",
    term: "DALL-E",
    shortDefinition: "Sistema de IA de OpenAI que genera imágenes a partir de texto.",
    fullDefinition: "Sistema de IA desarrollado por OpenAI que genera imágenes a partir de descripciones textuales, utilizando una variante de GPT-3 adaptada para comprender relaciones entre conceptos visuales.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["openai", "gpt", "difusion"],
    tags: ["openai", "imágenes", "texto-a-imagen"]
  },
  {
    id: "embedding",
    term: "Embeddings",
    shortDefinition: "Representaciones vectoriales de datos donde la proximidad indica similitud.",
    fullDefinition: "Representaciones vectoriales de datos (palabras, frases, imágenes) en un espacio multidimensional donde la proximidad indica similitud semántica.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["codificador", "nlp", "espacio-latente"],
    tags: ["representación", "vector", "semántica"]
  },
  {
    id: "evaluacion-automatica",
    term: "Evaluación Automática",
    shortDefinition: "Uso de software para juzgar la calidad del resultado de un modelo de IA.",
    fullDefinition: "Proceso que utiliza software para juzgar la calidad del resultado de un modelo de IA generativa, comparándolo con respuestas ideales o utilizando métricas predefinidas.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["evaluacion-evaluador-automatico", "rlhf"],
    tags: ["evaluación", "métricas"]
  },
  {
    id: "evaluacion-evaluador-automatico",
    term: "Evaluación del Evaluador Automático",
    shortDefinition: "Mecanismo híbrido que combina evaluación humana y automática para juzgar modelos.",
    fullDefinition: "Mecanismo híbrido para juzgar la calidad del resultado de un modelo de IA generativa que combina la evaluación humana con la evaluación automática.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["evaluacion-automatica", "rlhf"],
    tags: ["evaluación", "híbrido"]
  },
  {
    id: "fine-tuning",
    term: "Fine-tuning (Ajuste Fino)",
    shortDefinition: "Adaptar un modelo preentrenado a una tarea específica con datos adicionales.",
    fullDefinition: "Proceso de adaptar un modelo preentrenado a una tarea específica mediante entrenamiento adicional con un conjunto de datos más pequeño y especializado.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["transfer-learning", "modelo-base", "destilacion"],
    tags: ["entrenamiento", "especialización"]
  },
  {
    id: "funcion-de-perdida",
    term: "Función de Pérdida (Loss Function)",
    shortDefinition: "Medida matemática de la diferencia entre predicciones y valores reales.",
    fullDefinition: "Medida matemática que cuantifica la diferencia entre las predicciones de un modelo y los valores reales deseados durante el entrenamiento.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["gradiente-descendente", "entrenamiento"],
    tags: ["matemáticas", "optimización"]
  },
  {
    id: "gans",
    term: "GANs (Redes Generativas Adversarias)",
    shortDefinition: "Arquitectura con un generador y un discriminador que compiten para mejorar.",
    fullDefinition: "Arquitectura que enfrenta dos redes neuronales: un generador que crea contenido y un discriminador que evalúa su autenticidad, mejorando iterativamente.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["generador", "discriminador", "aprendizaje-profundo"],
    tags: ["modelo", "adversario", "generación"]
  },
  {
    id: "gemini",
    term: "Gemini",
    shortDefinition: "Modelo multimodal de Google DeepMind para procesar texto, imágenes, audio y video.",
    fullDefinition: "Modelo multimodal desarrollado por Google DeepMind, diseñado para comprender y generar contenido en múltiples formatos (texto, imágenes, audio, video).",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["google", "modelo-multimodal", "llm"],
    tags: ["google", "multimodal", "modelo"]
  },
  {
    id: "generacion-condicionada",
    term: "Generación Condicionada",
    shortDefinition: "Proceso generativo guiado por condiciones o restricciones específicas.",
    fullDefinition: "Técnica donde el proceso generativo se guía mediante condiciones o restricciones específicas proporcionadas como entrada, como descripciones textuales para imágenes.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["difusion", "prompt-engineering"],
    tags: ["generación", "control"]
  },
  {
    id: "ia-generativa",
    term: "GenAI (IA Generativa)",
    shortDefinition: "Subcampo de la IA enfocado en crear contenido nuevo a partir de datos existentes.",
    fullDefinition: "Subcampo de la inteligencia artificial que se enfoca en crear contenido nuevo a partir de datos existentes, pudiendo producir texto, imágenes, música y más de manera creativa y autónoma.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["algoritmo-generativo", "llm", "difusion"],
    tags: ["core", "ia"]
  },
  {
    id: "gpt",
    term: "GPT (Generative Pre-trained Transformer)",
    shortDefinition: "Familia de modelos de lenguaje de OpenAI basados en la arquitectura Transformer.",
    fullDefinition: "Familia de modelos de lenguaje desarrollados por OpenAI que utilizan la arquitectura transformer con un enfoque autoregresivo para generar texto.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["openai", "arquitectura-transformer", "llm", "chatgpt"],
    tags: ["openai", "modelo", "nlp"]
  },
  {
    id: "gradiente-descendente",
    term: "Gradiente Descendente",
    shortDefinition: "Algoritmo de optimización para entrenar redes neuronales ajustando parámetros.",
    fullDefinition: "Algoritmo de optimización utilizado para entrenar redes neuronales, que ajusta iterativamente los parámetros del modelo en la dirección que reduce la función de pérdida.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["funcion-de-perdida", "entrenamiento", "red-neuronal"],
    tags: ["optimización", "entrenamiento"]
  },
  {
    id: "hallucination",
    term: "Hallucination (Alucinación)",
    shortDefinition: "Fenómeno donde un modelo genera contenido plausible pero incorrecto o inventado.",
    fullDefinition: "Fenómeno donde un modelo generativo produce contenido que parece plausible pero es factualmente incorrecto o inventado.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["facticidad", "rag", "prompt-engineering"],
    tags: ["riesgo", "precisión"]
  },
  {
    id: "hiperparametros",
    term: "Hiperparámetros",
    shortDefinition: "Variables configurables que determinan la estructura y entrenamiento del modelo.",
    fullDefinition: "Variables configurables que determinan la estructura y el proceso de entrenamiento de un modelo, como la tasa de aprendizaje o el tamaño de lote.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["entrenamiento", "fine-tuning"],
    tags: ["configuración", "entrenamiento"]
  },
  {
    id: "in-context-learning",
    term: "In-context Learning (Aprendizaje en Contexto)",
    shortDefinition: "Capacidad de los LLM para adaptar su comportamiento basado en ejemplos en el prompt.",
    fullDefinition: "Capacidad de los modelos de lenguaje grandes para adaptar su comportamiento basándose en ejemplos proporcionados dentro del prompt, sin actualizar sus parámetros.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["few-shot-learning", "zero-shot-learning", "prompt-engineering"],
    tags: ["prompting", "adaptación"]
  },
  {
    id: "inferencia",
    term: "Inferencia",
    shortDefinition: "Proceso de usar un modelo entrenado para generar predicciones o contenido.",
    fullDefinition: "Proceso de utilizar un modelo entrenado para generar predicciones o contenido nuevo a partir de entradas específicas.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["entrenamiento", "modelo"],
    tags: ["proceso", "predicción"]
  },
  {
    id: "few-shot-learning",
    term: "Instrucción con Varios Ejemplos (Few-shot Learning)",
    shortDefinition: "Proporcionar al modelo algunos ejemplos en el prompt para guiar su tarea.",
    fullDefinition: "Técnica donde se proporciona al modelo algunos ejemplos dentro del prompt para guiar su comportamiento en una tarea específica.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["in-context-learning", "zero-shot-learning", "one-shot-learning"],
    tags: ["prompting", "aprendizaje"]
  },
  {
    id: "zero-shot-learning",
    term: "Instrucciones Directas (Zero-shot Learning)",
    shortDefinition: "Pedir al modelo realizar una tarea sin ejemplos previos, usando su conocimiento general.",
    fullDefinition: "Técnica donde se le pide al modelo realizar una tarea sin proporcionarle ejemplos previos, confiando en su conocimiento general.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["in-context-learning", "few-shot-learning", "modelo-base"],
    tags: ["prompting", "aprendizaje"]
  },
  {
    id: "espacio-latente",
    term: "Latent Space (Espacio Latente)",
    shortDefinition: "Representación comprimida y abstracta de datos dentro de un modelo generativo.",
    fullDefinition: "Representación comprimida y abstracta de datos dentro de un modelo generativo, donde cada punto corresponde a una posible salida.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["embedding", "difusion", "vae"],
    tags: ["representación", "abstracción"]
  },
  {
    id: "langchain",
    term: "LangChain",
    shortDefinition: "Framework para desarrollar aplicaciones potenciadas por LLMs.",
    fullDefinition: "Framework para desarrollar aplicaciones potenciadas por modelos de lenguaje, facilitando la conexión de LLMs con otras fuentes de datos y aplicaciones.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["llm", "llm-agents", "rag"],
    tags: ["framework", "desarrollo", "aplicaciones"]
  },
  {
    id: "llm",
    term: "LLM (Large Language Model)",
    shortDefinition: "Modelo de lenguaje con miles de millones de parámetros entrenado en vastos corpus.",
    fullDefinition: "Modelo de lenguaje con miles de millones o billones de parámetros, entrenado en vastos corpus de texto para adquirir capacidades lingüísticas y conocimientos generales.",
    category: "modelos",
    complexity: 1,
    relatedTerms: ["gpt", "bert", "gemini", "llama"],
    tags: ["modelo", "nlp", "core"]
  },
  {
    id: "llm-agents",
    term: "LLM Agents (Agentes LLM)",
    shortDefinition: "Sistemas que usan LLMs para interactuar con herramientas y ejecutar acciones.",
    fullDefinition: "Sistemas que utilizan modelos de lenguaje para interactuar con herramientas externas, tomar decisiones y ejecutar acciones en entornos digitales o físicos.",
    category: "aplicaciones",
    complexity: 3,
    relatedTerms: ["llm", "a2a", "manus", "langchain"],
    tags: ["agentes", "automatización", "aplicaciones"]
  },
  {
    id: "manus",
    term: "Manus",
    shortDefinition: "Agente de IA avanzado para realizar tareas complejas de forma autónoma.",
    fullDefinition: "Agente de IA avanzado diseñado para realizar tareas complejas de forma autónoma, utilizando herramientas y ejecutando flujos de trabajo completos con razonamiento sofisticado.",
    category: "aplicaciones",
    complexity: 3,
    relatedTerms: ["llm-agents", "a2a"],
    tags: ["agentes", "automatización", "google"]
  },
  {
    id: "memoria-de-contexto",
    term: "Memoria de Contexto",
    shortDefinition: "Capacidad de un modelo para mantener y usar información de interacciones previas.",
    fullDefinition: "Capacidad de un modelo para mantener y utilizar información de interacciones previas dentro de una conversación o sesión.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["ventana-de-contexto", "chatgpt"],
    tags: ["conversacional", "memoria"]
  },
  {
    id: "modelo-base",
    term: "Modelo Base",
    shortDefinition: "Versión inicial de un modelo entrenado en un corpus grande y diverso.",
    fullDefinition: "Versión inicial de un modelo de IA entrenado en un corpus grande y diverso, diseñado para capturar conocimientos generales antes de cualquier especialización.",
    category: "modelos",
    complexity: 1,
    relatedTerms: ["fine-tuning", "transfer-learning", "llm"],
    tags: ["modelo", "entrenamiento"]
  },
  {
    id: "modelo-multimodal",
    term: "Modelo Multimodal",
    shortDefinition: "Sistema de IA capaz de procesar y generar múltiples tipos de datos (texto, imagen, etc.).",
    fullDefinition: "Sistema de IA capaz de procesar y generar múltiples tipos de datos (texto, imágenes, audio, video) de manera integrada.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["gemini", "sora", "multimodal-chain-of-thought"],
    tags: ["modelo", "multimodal"]
  },
  {
    id: "moe",
    term: "MoE (Mixture of Experts)",
    shortDefinition: "Arquitectura que divide un modelo grande en subcomponentes especializados.",
    fullDefinition: "Arquitectura que divide un modelo grande en subcomponentes especializados (\"expertos\") que se activan selectivamente según la entrada.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["llm", "eficiencia"],
    tags: ["arquitectura", "eficiencia"]
  },
  {
    id: "multimodal-chain-of-thought",
    term: "Multimodal Chain-of-Thought",
    shortDefinition: "Extensión del razonamiento de cadena de pensamiento a contextos multimodales.",
    fullDefinition: "Extensión del razonamiento de cadena de pensamiento a contextos que involucran múltiples modalidades de información (texto, imágenes, etc.).",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["cadena-de-pensamientos", "modelo-multimodal"],
    tags: ["razonamiento", "multimodal"]
  },
  {
    id: "one-shot-learning",
    term: "One-shot Learning",
    shortDefinition: "Capacidad de un modelo para aprender una tarea nueva a partir de un solo ejemplo.",
    fullDefinition: "Capacidad de un modelo para aprender a realizar una tarea nueva a partir de un solo ejemplo, en contraste con el aprendizaje tradicional que requiere muchos ejemplos.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["few-shot-learning", "zero-shot-learning", "in-context-learning"],
    tags: ["aprendizaje", "prompting"]
  },
  {
    id: "overfitting",
    term: "Overfitting (Sobreajuste)",
    shortDefinition: "Modelo que aprende demasiado los datos de entrenamiento y pierde generalización.",
    fullDefinition: "Fenómeno donde un modelo aprende patrones específicos de los datos de entrenamiento tan detalladamente que pierde capacidad de generalización a datos nuevos.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["underfitting", "regularizacion", "weight-decay"],
    tags: ["entrenamiento", "riesgo"]
  },
  {
    id: "parametros",
    term: "Parámetros",
    shortDefinition: "Variables ajustables dentro de un modelo que se modifican durante el entrenamiento.",
    fullDefinition: "Variables ajustables dentro de un modelo de IA que se modifican durante el entrenamiento para capturar patrones en los datos.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["hiperparametros", "entrenamiento", "red-neuronal"],
    tags: ["modelo", "entrenamiento"]
  },
  {
    id: "perplexity",
    term: "Perplexity (Perplejidad)",
    shortDefinition: "Medida de qué tan bien un modelo de lenguaje predice una muestra de texto.",
    fullDefinition: "Medida matemática que evalúa qué tan bien un modelo de lenguaje predice una muestra de texto, con valores más bajos indicando mejor rendimiento.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["evaluacion-automatica", "llm"],
    tags: ["evaluación", "métricas", "nlp"]
  },
  {
    id: "prompt-engineering",
    term: "Prompt Engineering",
    shortDefinition: "Diseño y optimización de instrucciones textuales para modelos generativos.",
    fullDefinition: "Práctica de diseñar y optimizar instrucciones textuales para modelos generativos con el fin de obtener resultados específicos y de alta calidad.",
    category: "tecnicas",
    complexity: 1,
    relatedTerms: ["cadena-de-pensamientos", "in-context-learning", "prompt"],
    tags: ["interacción", "optimización"]
  },
  {
    id: "rag",
    term: "RAG (Retrieval-Augmented Generation)",
    shortDefinition: "Combina generación de contenido con recuperación de información externa.",
    fullDefinition: "Técnica que combina la generación de contenido con la recuperación de información de fuentes externas para mejorar la precisión y reducir alucinaciones.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["hallucination", "facticidad", "llm", "langchain"],
    tags: ["precisión", "recuperación", "generación"]
  },
  {
    id: "red-neuronal",
    term: "Red Neuronal",
    shortDefinition: "Sistema computacional inspirado en el cerebro, con nodos interconectados en capas.",
    fullDefinition: "Sistema computacional inspirado en la estructura del cerebro humano, compuesto por nodos interconectados (neuronas artificiales) organizados en capas.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["aprendizaje-profundo", "parametros", "arquitectura-transformer"],
    tags: ["core", "arquitectura"]
  },
  {
    id: "rlhf",
    term: "RLHF (Reinforcement Learning from Human Feedback)",
    shortDefinition: "Usa evaluaciones humanas para refinar el comportamiento de modelos generativos.",
    fullDefinition: "Técnica que utiliza evaluaciones humanas para refinar el comportamiento de modelos generativos, alineando sus salidas con las expectativas humanas.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["alineacion-ia", "fine-tuning", "evaluacion-automatica"],
    tags: ["entrenamiento", "alineación", "feedback"]
  },
  {
    id: "sampling",
    term: "Sampling (Muestreo)",
    shortDefinition: "Proceso de seleccionar el siguiente token durante la generación de texto.",
    fullDefinition: "Proceso de seleccionar la siguiente palabra o token durante la generación de texto, utilizando diferentes estrategias como temperatura o top-k.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["temperatura", "top-k", "generacion-condicionada"],
    tags: ["generación", "probabilidad"]
  },
  {
    id: "stable-diffusion",
    term: "Stable Diffusion",
    shortDefinition: "Modelo de difusión latente de código abierto para generar imágenes desde texto.",
    fullDefinition: "Modelo de difusión latente de código abierto para generación de imágenes a partir de descripciones textuales, democratizando el acceso a esta tecnología.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["difusion", "imagen-latente", "texto-a-imagen"],
    tags: ["modelo", "imágenes", "código-abierto"]
  },
  {
    id: "temperatura",
    term: "Temperatura",
    shortDefinition: "Hiperparámetro que controla la aleatoriedad en la generación de contenido.",
    fullDefinition: "Hiperparámetro que controla la aleatoriedad en la generación de contenido, con valores más altos produciendo resultados más diversos pero potencialmente menos coherentes.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["sampling", "hiperparametros", "generacion-condicionada"],
    tags: ["generación", "control", "aleatoriedad"]
  },
  {
    id: "tokenizacion",
    term: "Tokenización",
    shortDefinition: "Proceso de dividir texto en unidades más pequeñas (tokens) para el modelo.",
    fullDefinition: "Proceso de dividir texto en unidades más pequeñas (tokens) que el modelo puede procesar, pudiendo ser palabras, partes de palabras o caracteres.",
    category: "tecnicas",
    complexity: 1,
    relatedTerms: ["token", "ventana-de-contexto", "nlp"],
    tags: ["nlp", "procesamiento"]
  },
  {
    id: "transfer-learning",
    term: "Transfer Learning",
    shortDefinition: "Reutilizar un modelo entrenado para una tarea como punto de partida para otra.",
    fullDefinition: "Técnica donde un modelo entrenado para una tarea se reutiliza como punto de partida para una tarea diferente pero relacionada.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["fine-tuning", "modelo-base"],
    tags: ["entrenamiento", "eficiencia"]
  },
  {
    id: "underfitting",
    term: "Underfitting (Subajuste)",
    shortDefinition: "Modelo demasiado simple para capturar la complejidad de los datos.",
    fullDefinition: "Situación donde un modelo es demasiado simple para capturar la complejidad de los datos, resultando en un rendimiento deficiente.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["overfitting", "entrenamiento"],
    tags: ["entrenamiento", "riesgo"]
  },
  {
    id: "vae",
    term: "VAE (Variational Autoencoder)",
    shortDefinition: "Modelo generativo que codifica datos en una distribución probabilística latente.",
    fullDefinition: "Tipo de modelo generativo que aprende a codificar datos en una distribución probabilística en un espacio latente y luego decodificar muestras.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["espacio-latente", "codificador", "decodificador"],
    tags: ["modelo", "generación", "probabilidad"]
  },
  {
    id: "ventana-de-contexto",
    term: "Ventana de Contexto",
    shortDefinition: "Cantidad máxima de tokens que un modelo puede procesar simultáneamente.",
    fullDefinition: "Cantidad máxima de tokens que un modelo puede procesar simultáneamente, limitando la cantidad de información que puede considerar al generar respuestas.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["token", "llm", "memoria-de-contexto"],
    tags: ["modelo", "limitación"]
  },
  {
    id: "weight-decay",
    term: "Weight Decay (Decaimiento de Pesos)",
    shortDefinition: "Técnica de regularización que penaliza parámetros grandes para evitar sobreajuste.",
    fullDefinition: "Técnica de regularización que penaliza valores grandes en los parámetros del modelo durante el entrenamiento, ayudando a prevenir el sobreajuste.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["overfitting", "regularizacion", "entrenamiento"],
    tags: ["entrenamiento", "optimización"]
  },
  {
    id: "a2a",
    term: "A2A (Agent-to-Agent)",
    shortDefinition: "Paradigma de comunicación donde múltiples agentes de IA interactúan entre sí.",
    fullDefinition: "Paradigma de comunicación donde múltiples agentes de IA interactúan entre sí para resolver problemas complejos o colaborar en tareas.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["llm-agents", "manus"],
    tags: ["agentes", "colaboración"]
  },
  {
    id: "destilacion",
    term: "Destilación",
    shortDefinition: "Reducir el tamaño de un modelo grande (profesor) a uno más pequeño (estudiante).",
    fullDefinition: "Proceso de reducir el tamaño de un modelo (profesor) en un modelo más pequeño (estudiante) que emula las predicciones del original de manera más eficiente.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["fine-tuning", "eficiencia"],
    tags: ["optimización", "eficiencia"]
  },
  {
    id: "facticidad",
    term: "Facticidad",
    shortDefinition: "Propiedad de un modelo cuyo resultado se basa en la realidad y hechos verificables.",
    fullDefinition: "Propiedad que describe un modelo cuyo resultado se basa en la realidad y hechos verificables, en contraste con la creatividad pura.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["hallucination", "rag"],
    tags: ["precisión", "confiabilidad"]
  },
  {
    id: "mcp",
    term: "MCP (Model Control Protocol)",
    shortDefinition: "Protocolo estándar para la comunicación entre aplicaciones y modelos de IA.",
    fullDefinition: "Protocolo estándar que define cómo las aplicaciones pueden comunicarse con modelos de IA, permitiendo una integración consistente y flexible entre diferentes sistemas y modelos.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["llm", "llm-agents", "api"],
    tags: ["integración", "protocolo", "estándar"]
  },
  {
    id: "ag-ui",
    term: "AG-UI (Agent User Interface)",
    shortDefinition: "Interfaz diseñada específicamente para la interacción con agentes de IA.",
    fullDefinition: "Interfaz de usuario especializada para facilitar la interacción entre humanos y agentes de IA, optimizada para mostrar el razonamiento, acciones y resultados del agente de manera comprensible.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["llm-agents", "manus", "ui-ux"],
    tags: ["interfaz", "agentes", "interacción"]
  },
  {
    id: "alineacion-ia",
    term: "Alineación de IA",
    shortDefinition: "Proceso de asegurar que los sistemas de IA actúen según valores e intenciones humanas.",
    fullDefinition: "Campo de investigación y conjunto de técnicas enfocadas en asegurar que los sistemas de IA actúen de acuerdo con los valores e intenciones humanas, abordando problemas de seguridad y ética.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["rlhf", "ia-segura", "etica-ia"],
    tags: ["seguridad", "ética", "valores"]
  },
  {
    id: "ia-segura",
    term: "IA Segura",
    shortDefinition: "Enfoque para desarrollar sistemas de IA que minimicen riesgos y daños potenciales.",
    fullDefinition: "Enfoque para el desarrollo de sistemas de IA que busca minimizar riesgos y daños potenciales, asegurando que los sistemas sean robustos, confiables y alineados con valores humanos.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["alineacion-ia", "etica-ia", "rlhf"],
    tags: ["seguridad", "riesgo", "robustez"]
  },
  {
    id: "etica-ia",
    term: "Ética en IA",
    shortDefinition: "Principios morales y consideraciones éticas aplicadas al desarrollo y uso de IA.",
    fullDefinition: "Campo que estudia los principios morales y consideraciones éticas aplicadas al desarrollo, implementación y uso de sistemas de inteligencia artificial.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["alineacion-ia", "ia-segura", "sesgo-algoritmico"],
    tags: ["ética", "valores", "responsabilidad"]
  },
  {
    id: "sesgo-algoritmico",
    term: "Sesgo Algorítmico",
    shortDefinition: "Tendencia sistemática de un algoritmo a favorecer ciertos resultados injustamente.",
    fullDefinition: "Tendencia sistemática de un algoritmo o modelo de IA a favorecer ciertos resultados o grupos sobre otros de manera injusta, reflejando y potencialmente amplificando sesgos presentes en los datos de entrenamiento.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["etica-ia", "equidad-algoritmica", "datos-entrenamiento"],
    tags: ["ética", "equidad", "riesgo"]
  },
  {
    id: "equidad-algoritmica",
    term: "Equidad Algorítmica",
    shortDefinition: "Principio de diseñar algoritmos que traten a todos los grupos de manera justa.",
    fullDefinition: "Principio y conjunto de técnicas para diseñar algoritmos y sistemas de IA que traten a todos los grupos demográficos de manera justa, evitando discriminación y sesgos injustos.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["sesgo-algoritmico", "etica-ia", "alineacion-ia"],
    tags: ["ética", "equidad", "justicia"]
  },
  {
    id: "api",
    term: "API (Application Programming Interface)",
    shortDefinition: "Conjunto de reglas que permite a diferentes aplicaciones comunicarse entre sí.",
    fullDefinition: "Conjunto de reglas, protocolos y herramientas que permite a diferentes aplicaciones de software comunicarse entre sí, facilitando la integración de servicios de IA en aplicaciones existentes.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["mcp", "rest-api", "integracion"],
    tags: ["desarrollo", "integración", "comunicación"]
  },
  {
    id: "rest-api",
    term: "REST API",
    shortDefinition: "Estilo arquitectónico para diseñar servicios web basados en HTTP.",
    fullDefinition: "Estilo arquitectónico para diseñar servicios web que utiliza métodos HTTP estándar y es ampliamente utilizado para crear APIs que permiten acceder a servicios de IA.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["api", "http", "integracion"],
    tags: ["desarrollo", "web", "comunicación"]
  },
  {
    id: "integracion",
    term: "Integración de IA",
    shortDefinition: "Proceso de incorporar capacidades de IA en aplicaciones y sistemas existentes.",
    fullDefinition: "Proceso de incorporar capacidades de inteligencia artificial en aplicaciones, productos y sistemas existentes para mejorar su funcionalidad y crear nuevas experiencias de usuario.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["api", "mcp", "llm-agents"],
    tags: ["desarrollo", "implementación", "aplicaciones"]
  },
  {
    id: "ui-ux",
    term: "UI/UX para IA",
    shortDefinition: "Diseño de interfaces y experiencias de usuario específicas para sistemas de IA.",
    fullDefinition: "Disciplina que se enfoca en el diseño de interfaces y experiencias de usuario específicamente adaptadas para sistemas de IA, considerando aspectos como transparencia, control del usuario y comunicación de capacidades y limitaciones.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["ag-ui", "transparencia-ia", "interaccion-humano-ia"],
    tags: ["diseño", "interfaz", "experiencia"]
  },
  {
    id: "transparencia-ia",
    term: "Transparencia en IA",
    shortDefinition: "Principio de hacer comprensibles los procesos y decisiones de los sistemas de IA.",
    fullDefinition: "Principio que busca hacer comprensibles y explicables los procesos internos y decisiones de los sistemas de IA para usuarios, desarrolladores y reguladores.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["explicabilidad", "ui-ux", "etica-ia"],
    tags: ["ética", "confianza", "explicabilidad"]
  },
  {
    id: "explicabilidad",
    term: "Explicabilidad (XAI)",
    shortDefinition: "Capacidad de un sistema de IA para explicar sus decisiones de manera comprensible.",
    fullDefinition: "Capacidad de un sistema de inteligencia artificial para explicar sus decisiones, predicciones o comportamientos de manera comprensible para los humanos, facilitando la confianza y la supervisión.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["transparencia-ia", "interpretabilidad", "caja-negra"],
    tags: ["transparencia", "confianza", "interpretación"]
  },
  {
    id: "interpretabilidad",
    term: "Interpretabilidad",
    shortDefinition: "Grado en que un humano puede entender la causa de una decisión de IA.",
    fullDefinition: "Grado en que un humano puede entender la causa de una decisión tomada por un modelo de IA, relacionado con la transparencia del modelo y sus mecanismos internos.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["explicabilidad", "transparencia-ia", "caja-negra"],
    tags: ["transparencia", "comprensión", "análisis"]
  },
  {
    id: "caja-negra",
    term: "Caja Negra (Black Box)",
    shortDefinition: "Sistema de IA cuyo funcionamiento interno es difícil o imposible de interpretar.",
    fullDefinition: "Sistema de inteligencia artificial cuyo funcionamiento interno es difícil o imposible de interpretar debido a su complejidad, opacidad o propiedad intelectual protegida.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["explicabilidad", "interpretabilidad", "transparencia-ia"],
    tags: ["opacidad", "complejidad", "riesgo"]
  },
  {
    id: "interaccion-humano-ia",
    term: "Interacción Humano-IA",
    shortDefinition: "Estudio de cómo los humanos y los sistemas de IA se comunican y colaboran.",
    fullDefinition: "Campo interdisciplinario que estudia cómo los humanos y los sistemas de inteligencia artificial se comunican, colaboran y coexisten, buscando optimizar esta relación para beneficio mutuo.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["ui-ux", "ag-ui", "colaboracion-humano-ia"],
    tags: ["interacción", "colaboración", "diseño"]
  },
  {
    id: "colaboracion-humano-ia",
    term: "Colaboración Humano-IA",
    shortDefinition: "Enfoque donde humanos y sistemas de IA trabajan juntos complementando capacidades.",
    fullDefinition: "Enfoque donde humanos y sistemas de inteligencia artificial trabajan juntos en tareas complejas, complementando sus respectivas capacidades y limitaciones para lograr mejores resultados que por separado.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["interaccion-humano-ia", "augmented-intelligence", "ia-centrada-humano"],
    tags: ["colaboración", "sinergia", "trabajo"]
  },
  {
    id: "augmented-intelligence",
    term: "Inteligencia Aumentada",
    shortDefinition: "Uso de IA para mejorar las capacidades humanas en lugar de reemplazarlas.",
    fullDefinition: "Enfoque que utiliza la inteligencia artificial para mejorar y amplificar las capacidades cognitivas humanas en lugar de reemplazarlas, enfatizando la colaboración entre humanos y máquinas.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["colaboracion-humano-ia", "ia-centrada-humano", "interaccion-humano-ia"],
    tags: ["colaboración", "potenciación", "asistencia"]
  },
  {
    id: "ia-centrada-humano",
    term: "IA Centrada en el Humano",
    shortDefinition: "Enfoque de diseño de IA que prioriza las necesidades y valores humanos.",
    fullDefinition: "Filosofía y metodología de diseño de sistemas de IA que prioriza las necesidades, valores y bienestar humanos, asegurando que la tecnología sirva y empodere a las personas.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["etica-ia", "augmented-intelligence", "colaboracion-humano-ia"],
    tags: ["diseño", "valores", "humanismo"]
  },
  {
    id: "llama",
    term: "Llama",
    shortDefinition: "Familia de modelos de lenguaje de código abierto desarrollados por Meta.",
    fullDefinition: "Familia de modelos de lenguaje de gran escala y código abierto desarrollados por Meta (anteriormente Facebook), diseñados para ser eficientes y accesibles para investigación y aplicaciones.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["llm", "meta", "codigo-abierto"],
    tags: ["modelo", "código-abierto", "meta"]
  },
  {
    id: "codigo-abierto",
    term: "IA de Código Abierto",
    shortDefinition: "Modelos y herramientas de IA disponibles públicamente con licencias permisivas.",
    fullDefinition: "Modelos, herramientas y frameworks de inteligencia artificial que están disponibles públicamente con licencias que permiten su uso, modificación y distribución libre, democratizando el acceso a la tecnología.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["llama", "stable-diffusion", "huggingface"],
    tags: ["accesibilidad", "comunidad", "democratización"]
  },
  {
    id: "huggingface",
    term: "Hugging Face",
    shortDefinition: "Plataforma y comunidad para compartir modelos de IA y herramientas de NLP.",
    fullDefinition: "Plataforma y comunidad que facilita el desarrollo y compartición de modelos de IA, especialmente en procesamiento de lenguaje natural, con miles de modelos preentrenados disponibles públicamente.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["codigo-abierto", "transformers", "nlp"],
    tags: ["plataforma", "comunidad", "recursos"]
  },
  {
    id: "transformers",
    term: "Transformers (Biblioteca)",
    shortDefinition: "Biblioteca de código abierto para trabajar con modelos basados en transformers.",
    fullDefinition: "Biblioteca de código abierto desarrollada por Hugging Face que proporciona APIs para trabajar con modelos basados en la arquitectura transformer, facilitando su uso en diversas aplicaciones.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["huggingface", "arquitectura-transformer", "codigo-abierto"],
    tags: ["biblioteca", "desarrollo", "herramienta"]
  },
  {
    id: "sora",
    term: "Sora",
    shortDefinition: "Modelo de OpenAI para generar videos realistas a partir de descripciones textuales.",
    fullDefinition: "Modelo de inteligencia artificial desarrollado por OpenAI capaz de generar videos realistas y creativos a partir de descripciones textuales, representando un avance significativo en la generación de contenido multimodal.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["openai", "modelo-multimodal", "texto-a-video"],
    tags: ["video", "generación", "multimodal"]
  },
  {
    id: "texto-a-video",
    term: "Texto a Video",
    shortDefinition: "Tecnología que genera contenido de video basado en descripciones textuales.",
    fullDefinition: "Tecnología de IA que genera contenido de video basado en descripciones textuales, permitiendo crear secuencias animadas a partir de instrucciones en lenguaje natural.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["sora", "generacion-condicionada", "modelo-multimodal"],
    tags: ["generación", "video", "multimodal"]
  },
  {
    id: "texto-a-imagen",
    term: "Texto a Imagen",
    shortDefinition: "Tecnología que genera imágenes basadas en descripciones textuales.",
    fullDefinition: "Tecnología de IA que genera imágenes basadas en descripciones textuales, permitiendo crear representaciones visuales a partir de instrucciones en lenguaje natural.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["dall-e", "stable-diffusion", "difusion"],
    tags: ["generación", "imágenes", "multimodal"]
  },
  {
    id: "imagen-latente",
    term: "Imagen Latente",
    shortDefinition: "Representación codificada de una imagen en el espacio latente de un modelo.",
    fullDefinition: "Representación codificada de una imagen en el espacio latente de un modelo generativo, que captura características esenciales y puede ser manipulada para generar variaciones.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["espacio-latente", "difusion", "vae"],
    tags: ["representación", "generación", "manipulación"]
  },
  {
    id: "token",
    term: "Token",
    shortDefinition: "Unidad básica de procesamiento en modelos de lenguaje (palabra, subpalabra o carácter).",
    fullDefinition: "Unidad básica de procesamiento en modelos de lenguaje, que puede ser una palabra completa, parte de una palabra o un carácter individual, dependiendo del esquema de tokenización utilizado.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["tokenizacion", "ventana-de-contexto", "llm"],
    tags: ["nlp", "procesamiento", "unidad"]
  },
  {
    id: "top-k",
    term: "Top-k Sampling",
    shortDefinition: "Técnica que limita la selección de tokens a los k más probables durante la generación.",
    fullDefinition: "Técnica de muestreo que limita la selección del siguiente token a los k más probables durante la generación de texto, ayudando a controlar la calidad y relevancia del contenido generado.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["sampling", "temperatura", "generacion-condicionada"],
    tags: ["generación", "probabilidad", "control"]
  },
  {
    id: "regularizacion",
    term: "Regularización",
    shortDefinition: "Técnicas para prevenir el sobreajuste en modelos de aprendizaje automático.",
    fullDefinition: "Conjunto de técnicas utilizadas durante el entrenamiento de modelos de aprendizaje automático para prevenir el sobreajuste y mejorar la generalización a datos nuevos.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["overfitting", "weight-decay", "dropout"],
    tags: ["entrenamiento", "generalización", "optimización"]
  },
  {
    id: "dropout",
    term: "Dropout",
    shortDefinition: "Técnica de regularización que desactiva aleatoriamente neuronas durante el entrenamiento.",
    fullDefinition: "Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante cada paso de entrenamiento, forzando a la red a aprender representaciones más robustas y redundantes.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["regularizacion", "overfitting", "red-neuronal"],
    tags: ["entrenamiento", "regularización", "redes"]
  },
    {
    id: "veo3",
    term: "VEO3",
    shortDefinition: "Plataforma de inteligencia artificial para procesamiento y análisis de datos.",
    fullDefinition: "Plataforma avanzada de inteligencia artificial diseñada para el procesamiento, análisis y visualización de grandes volúmenes de datos, con capacidades de integración con diversos sistemas y fuentes de información.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["ia-generativa", "procesamiento-datos"],
    tags: ["plataforma", "análisis", "datos"]
  },
  {
    id: "veo2",
    term: "VEO2",
    shortDefinition: "Versión anterior de la plataforma VEO para análisis de datos.",
    fullDefinition: "Versión previa de la plataforma VEO que ofrece capacidades de análisis de datos e inteligencia artificial, con enfoque en la integración de sistemas y procesamiento de información estructurada.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["veo3", "procesamiento-datos"],
    tags: ["plataforma", "análisis", "datos"]
  },
  {
    id: "ollama",
    term: "Ollama",
    shortDefinition: "Herramienta para ejecutar modelos de lenguaje localmente.",
    fullDefinition: "Herramienta de código abierto que permite ejecutar modelos de lenguaje de gran tamaño (LLMs) localmente en dispositivos personales, facilitando el acceso a capacidades de IA sin depender de servicios en la nube.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["llm", "inferencia-local"],
    tags: ["local", "llm", "herramienta"]
  },
  {
    id: "deepseek",
    term: "DeepSeek",
    shortDefinition: "Familia de modelos de lenguaje de código abierto con capacidades multimodales.",
    fullDefinition: "Conjunto de modelos de lenguaje de código abierto desarrollados por DeepSeek AI, que ofrecen capacidades avanzadas de procesamiento de lenguaje natural y comprensión multimodal, incluyendo variantes especializadas para programación y tareas específicas.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["llm", "codigo-abierto", "modelo-multimodal"],
    tags: ["modelo", "código abierto", "multimodal"]
  },
  {
    id: "perplexity",
    term: "Perplexity",
    shortDefinition: "Motor de búsqueda potenciado por IA con capacidades de respuesta y citación.",
    fullDefinition: "Motor de búsqueda avanzado potenciado por inteligencia artificial que proporciona respuestas directas a consultas complejas, con capacidad para citar fuentes y ofrecer información actualizada mediante la integración de modelos de lenguaje con búsqueda en tiempo real.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["rag", "motor-busqueda", "llm"],
    tags: ["búsqueda", "respuestas", "citación"]
  },
  {
    id: "deepvoice",
    term: "DeepVoice",
    shortDefinition: "Tecnología de síntesis de voz basada en redes neuronales profundas.",
    fullDefinition: "Sistema de síntesis de voz que utiliza redes neuronales profundas para generar voces humanas realistas, permitiendo la conversión de texto a voz con entonación natural y expresividad.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["sintesis-voz", "redes-neuronales", "audio-generativo"],
    tags: ["audio", "voz", "síntesis"]
  },
  {
    id: "deepface",
    term: "DeepFace",
    shortDefinition: "Framework de reconocimiento facial basado en aprendizaje profundo.",
    fullDefinition: "Framework de código abierto para reconocimiento facial que utiliza técnicas de aprendizaje profundo para detectar, analizar y verificar rostros en imágenes, con capacidades de identificación de emociones y atributos faciales.",
    category: "aplicaciones",
    complexity: 3,
    relatedTerms: ["reconocimiento-facial", "vision-computacional", "aprendizaje-profundo"],
    tags: ["facial", "reconocimiento", "visión"]
  },
  {
    id: "whisper",
    term: "Whisper",
    shortDefinition: "Modelo de reconocimiento de voz multilingüe de OpenAI.",
    fullDefinition: "Sistema de reconocimiento automático de voz desarrollado por OpenAI, capaz de transcribir y traducir audio en múltiples idiomas con alta precisión, incluso en condiciones acústicas desafiantes.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["openai", "reconocimiento-voz", "transcripcion"],
    tags: ["audio", "transcripción", "multilingüe"]
  },
  {
    id: "google-notebooklm",
    term: "Google NotebookLM",
    shortDefinition: "Herramienta de Google que combina LLMs con documentos personales para crear asistentes personalizados.",
    fullDefinition: "Aplicación experimental de Google que permite a los usuarios cargar documentos personales para crear asistentes de IA personalizados, facilitando la interacción con el contenido mediante preguntas y respuestas contextualizadas basadas en los documentos proporcionados.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["google", "rag", "asistente-ia"],
    tags: ["google", "documentos", "personalización"]
  },
  {
    id: "graficadora",
    term: "Graficadora",
    shortDefinition: "Herramienta para visualización de datos y creación de gráficos.",
    fullDefinition: "Software especializado en la visualización de datos y creación de gráficos interactivos, permitiendo representar información compleja de manera visual para facilitar su análisis e interpretación.",
    category: "aplicaciones",
    complexity: 1,
    relatedTerms: ["visualizacion-datos", "analisis-datos"],
    tags: ["visualización", "gráficos", "datos"]
  },
  {
    id: "n8n",
    term: "n8n",
    shortDefinition: "Plataforma de automatización de flujos de trabajo de código abierto.",
    fullDefinition: "Plataforma de automatización de flujos de trabajo de código abierto que permite conectar diferentes aplicaciones y servicios para crear procesos automatizados sin necesidad de programación avanzada, con una interfaz visual para diseñar los flujos.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["automatizacion", "integracion", "flujo-trabajo"],
    tags: ["automatización", "workflow", "código abierto"]
  },
  {
    id: "flowise",
    term: "Flowise",
    shortDefinition: "Plataforma de código abierto para crear aplicaciones con LangChain mediante interfaz visual.",
    fullDefinition: "Herramienta de código abierto que proporciona una interfaz visual para construir aplicaciones basadas en LangChain, facilitando la creación de flujos de trabajo con modelos de lenguaje sin necesidad de escribir código.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["langchain", "llm", "automatizacion"],
    tags: ["desarrollo", "visual", "langchain"]
  },
  {
    id: "google-ai-studio",
    term: "Google AI Studio",
    shortDefinition: "Plataforma de Google para experimentar con modelos de IA y crear aplicaciones.",
    fullDefinition: "Entorno de desarrollo web creado por Google que permite a desarrolladores experimentar con modelos de IA como Gemini, crear prompts, ajustar parámetros y desarrollar aplicaciones basadas en inteligencia artificial.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["google", "gemini", "desarrollo-ia"],
    tags: ["google", "desarrollo", "experimentación"]
  },
  {
    id: "grok",
    term: "Grok",
    shortDefinition: "Modelo de lenguaje conversacional desarrollado por xAI.",
    fullDefinition: "Modelo de lenguaje conversacional desarrollado por xAI (empresa de Elon Musk), diseñado para proporcionar respuestas con un estilo más informal y humorístico, con acceso a información en tiempo real a través de navegación web.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["llm", "xai", "modelo-conversacional"],
    tags: ["xai", "conversacional", "tiempo real"]
  },
  {
    id: "serpapi",
    term: "SerpAPI",
    shortDefinition: "API para extraer resultados de motores de búsqueda de manera estructurada.",
    fullDefinition: "Servicio que proporciona una API para extraer resultados de motores de búsqueda como Google, Bing y otros, devolviendo datos estructurados que facilitan su procesamiento e integración en aplicaciones.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["api", "busqueda-web", "datos-estructurados"],
    tags: ["búsqueda", "api", "extracción"]
  },
  {
    id: "groq",
    term: "Groq",
    shortDefinition: "Plataforma de inferencia de IA optimizada para velocidad y eficiencia.",
    fullDefinition: "Empresa y plataforma de inferencia de IA que utiliza hardware especializado (LPU - Language Processing Unit) para ejecutar modelos de lenguaje con velocidad excepcional, ofreciendo tiempos de respuesta significativamente más rápidos que otras soluciones.",
    category: "empresas",
    complexity: 3,
    relatedTerms: ["inferencia", "llm", "aceleracion-hardware"],
    tags: ["velocidad", "inferencia", "hardware"]
  },
  {
    id: "openrouter",
    term: "OpenRouter",
    shortDefinition: "Servicio que proporciona acceso unificado a múltiples modelos de IA.",
    fullDefinition: "Plataforma que ofrece una API unificada para acceder a diversos modelos de lenguaje de diferentes proveedores, simplificando la integración y permitiendo cambiar fácilmente entre modelos sin modificar el código de la aplicación.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["api", "llm", "integracion-modelos"],
    tags: ["api", "integración", "múltiples modelos"]
  },
  {
    id: "firecrawl",
    term: "Firecrawl",
    shortDefinition: "Herramienta de rastreo web para recopilación de datos estructurados.",
    fullDefinition: "Herramienta especializada en rastreo web (web crawling) que permite recopilar datos estructurados de sitios web de manera eficiente, facilitando la creación de conjuntos de datos para entrenamiento o análisis.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["web-scraping", "datos-estructurados", "recopilacion-datos"],
    tags: ["web", "datos", "rastreo"]
  },
  {
    id: "napkin",
    term: "Napkin",
    shortDefinition: "Plataforma para crear aplicaciones de IA mediante instrucciones en lenguaje natural.",
    fullDefinition: "Plataforma que permite a usuarios crear aplicaciones de IA simplemente describiendo lo que quieren en lenguaje natural, traduciendo estas descripciones en aplicaciones funcionales sin necesidad de programación tradicional.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["desarrollo-sin-codigo", "llm", "aplicaciones-ia"],
    tags: ["desarrollo", "lenguaje natural", "sin código"]
  },
  {
    id: "openwebui",
    term: "OpenWebUI",
    shortDefinition: "Interfaz web de código abierto para interactuar con modelos de IA locales.",
    fullDefinition: "Interfaz web de código abierto diseñada para interactuar con modelos de lenguaje ejecutados localmente (como los de Ollama), proporcionando una experiencia similar a ChatGPT pero para modelos alojados en el propio dispositivo del usuario.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["ollama", "interfaz-usuario", "llm-local"],
    tags: ["interfaz", "local", "código abierto"]
  },
  {
    id: "qwen-2-5",
    term: "Qwen 2.5",
    shortDefinition: "Familia de modelos de lenguaje desarrollados por Alibaba.",
    fullDefinition: "Serie de modelos de lenguaje de gran tamaño desarrollados por Alibaba Cloud, que incluye variantes de diferentes tamaños optimizadas para diversas tareas, con capacidades multilingües y multimodales avanzadas.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["llm", "alibaba", "modelo-multimodal"],
    tags: ["alibaba", "multilingüe", "multimodal"]
  },
  {
    id: "sam-altman",
    term: "Sam Altman",
    shortDefinition: "CEO de OpenAI y figura influyente en el desarrollo de la IA.",
    fullDefinition: "Director ejecutivo de OpenAI y figura prominente en el campo de la inteligencia artificial, conocido por liderar el desarrollo de modelos como GPT y por su visión sobre el impacto futuro de la IA en la sociedad.",
    category: "empresas",
    complexity: 1,
    relatedTerms: ["openai", "gpt", "chatgpt"],
    tags: ["openai", "liderazgo", "industria"]
  },
  {
    id: "cuda",
    term: "CUDA",
    shortDefinition: "Plataforma de computación paralela de NVIDIA para procesamiento en GPUs.",
    fullDefinition: "Plataforma de computación paralela y modelo de programación desarrollado por NVIDIA que permite utilizar las unidades de procesamiento gráfico (GPUs) para cálculos de propósito general, acelerando significativamente tareas como el entrenamiento de modelos de aprendizaje profundo.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["gpu", "aceleracion-hardware", "entrenamiento-ia"],
    tags: ["nvidia", "gpu", "computación paralela"]
  },
  {
    id: "lora",
    term: "LoRA (Low-Rank Adaptation)",
    shortDefinition: "Técnica eficiente para ajustar modelos de lenguaje grandes.",
    fullDefinition: "Método de adaptación de bajo rango que permite ajustar modelos de lenguaje grandes de manera eficiente, reduciendo significativamente los requisitos de memoria y computación al modificar solo un pequeño conjunto de parámetros.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["fine-tuning", "llm", "parametros-eficientes"],
    tags: ["ajuste", "eficiencia", "parámetros"]
  },
  {
    id: "gpu",
    term: "GPU (Unidad de Procesamiento Gráfico)",
    shortDefinition: "Hardware especializado para procesamiento paralelo, crucial en IA.",
    fullDefinition: "Unidad de procesamiento gráfico, componente de hardware especializado en realizar cálculos en paralelo, fundamental para el entrenamiento e inferencia de modelos de aprendizaje profundo debido a su capacidad para procesar grandes cantidades de datos simultáneamente.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["cuda", "entrenamiento-ia", "aceleracion-hardware"],
    tags: ["hardware", "procesamiento", "paralelo"]
  },
  {
    id: "gema-de-gemini",
    term: "Gema de Gemini",
    shortDefinition: "Versión compacta y eficiente del modelo Gemini de Google.",
    fullDefinition: "Versión optimizada y más ligera del modelo Gemini de Google, diseñada para funcionar eficientemente en dispositivos con recursos limitados mientras mantiene un alto nivel de capacidad y rendimiento.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["gemini", "google", "modelo-eficiente"],
    tags: ["google", "eficiencia", "compacto"]
  },
  {
    id: "github",
    term: "GitHub",
    shortDefinition: "Plataforma de desarrollo colaborativo y control de versiones basada en Git.",
    fullDefinition: "Plataforma basada en la nube para desarrollo colaborativo de software que utiliza Git para control de versiones, permitiendo a desarrolladores almacenar, gestionar, rastrear y controlar cambios en su código, además de facilitar la colaboración en proyectos de código abierto y privados.",
    category: "empresas",
    complexity: 2,
    relatedTerms: ["git", "desarrollo-colaborativo", "codigo-abierto"],
    tags: ["desarrollo", "colaboración", "código"]
  },
  {
    id: "supabase",
    term: "Supabase",
    shortDefinition: "Alternativa de código abierto a Firebase con base de datos PostgreSQL.",
    fullDefinition: "Plataforma de desarrollo de código abierto que proporciona funcionalidades similares a Firebase, incluyendo base de datos PostgreSQL, autenticación, almacenamiento y funciones en tiempo real, permitiendo crear aplicaciones completas con infraestructura backend.",
    category: "empresas",
    complexity: 2,
    relatedTerms: ["postgresql", "backend", "desarrollo-aplicaciones"],
    tags: ["base de datos", "backend", "código abierto"]
  },
  {
    id: "vector",
    term: "Vector",
    shortDefinition: "Estructura de datos que representa magnitudes con dirección en un espacio multidimensional.",
    fullDefinition: "Estructura matemática y de datos que representa magnitudes con dirección en un espacio multidimensional, fundamental en IA para representar datos como embeddings, permitiendo operaciones como búsqueda de similitud y manipulación algebraica.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["embedding", "espacio-latente", "matriz"],
    tags: ["matemáticas", "representación", "datos"]
  },
  {
    id: "matriz",
    term: "Matriz",
    shortDefinition: "Arreglo bidimensional de números utilizado en álgebra lineal y procesamiento de datos.",
    fullDefinition: "Estructura de datos bidimensional organizada en filas y columnas, fundamental en álgebra lineal y ampliamente utilizada en IA para representar datos, pesos de redes neuronales y transformaciones lineales.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["algebra-lineal", "vector", "tensor"],
    tags: ["matemáticas", "álgebra", "datos"]
  },
  {
    id: "cursor",
    term: "Cursor",
    shortDefinition: "IDE de programación potenciado por IA para desarrollo de software.",
    fullDefinition: "Entorno de desarrollo integrado (IDE) potenciado por inteligencia artificial, diseñado para mejorar la productividad de los programadores mediante funciones como autocompletado avanzado, generación de código y asistencia contextual basada en modelos de lenguaje.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["ide", "desarrollo-software", "asistencia-codigo"],
    tags: ["programación", "ide", "asistencia"]
  },
  {
    id: "trae",
    term: "Trae",
    shortDefinition: "IDE de programación con capacidades de IA para asistencia en desarrollo.",
    fullDefinition: "Entorno de desarrollo integrado que incorpora capacidades de inteligencia artificial para asistir a los desarrolladores en la escritura, depuración y optimización de código, ofreciendo sugerencias contextuales y automatización de tareas repetitivas.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["ide", "desarrollo-software", "asistencia-codigo"],
    tags: ["programación", "ide", "asistencia"]
  },
  {
    id: "copilot",
    term: "Copilot",
    shortDefinition: "Asistente de programación de GitHub basado en IA que sugiere código.",
    fullDefinition: "Herramienta de asistencia de programación desarrollada por GitHub y OpenAI que utiliza modelos de lenguaje para sugerir código en tiempo real mientras el desarrollador escribe, capaz de generar funciones completas y bloques de código basados en comentarios y contexto.",
    category: "aplicaciones",
    complexity: 2,
    relatedTerms: ["github", "openai", "codex", "asistencia-codigo"],
    tags: ["programación", "asistencia", "generación de código"]
  },
  {
    id: "json",
    term: "JSON (JavaScript Object Notation)",
    shortDefinition: "Formato ligero de intercambio de datos basado en la sintaxis de JavaScript.",
    fullDefinition: "Formato de texto ligero para el intercambio de datos, basado en la sintaxis de objetos de JavaScript pero independiente del lenguaje, ampliamente utilizado en aplicaciones web y APIs para transmitir datos estructurados.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["api", "intercambio-datos", "javascript"],
    tags: ["formato", "datos", "web"]
  },
  {
    id: "codex-de-openai",
    term: "Codex de OpenAI",
    shortDefinition: "Modelo de IA especializado en generación y comprensión de código.",
    fullDefinition: "Modelo de inteligencia artificial desarrollado por OpenAI, derivado de GPT y especializado en la comprensión y generación de código en múltiples lenguajes de programación, que sirve como base para herramientas como GitHub Copilot.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["openai", "gpt", "copilot", "generacion-codigo"],
    tags: ["código", "programación", "openai"]
  },
  {
    id: "windsurf",
    term: "Windsurf",
    shortDefinition: "Deporte acuático que combina elementos de surf y vela.",
    fullDefinition: "Deporte acuático que combina elementos del surf y la navegación a vela, donde el participante se desplaza sobre una tabla propulsada por el viento a través de una vela articulada, permitiendo realizar maniobras y saltos sobre el agua.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["deporte-acuatico", "vela", "surf"],
    tags: ["deporte", "agua", "viento"]
  }

];
const categories = [
  { id: "modelos", name: "Modelos", color: "var(--cat-modelos)", description: "Arquitecturas y tipos específicos de sistemas de IA generativa." },
  { id: "tecnicas", name: "Técnicas", color: "var(--cat-tecnicas)", description: "Métodos y procesos utilizados en el desarrollo y aplicación de la IA generativa." },
  { id: "conceptos", name: "Conceptos", color: "var(--cat-conceptos)", description: "Ideas y principios fundamentales que sustentan la IA generativa." },
  { id: "aplicaciones", name: "Aplicaciones", color: "var(--cat-aplicaciones)", description: "Herramientas, plataformas y casos de uso específicos de la IA generativa." },
  { id: "empresas", name: "Empresas", color: "var(--cat-empresas)", description: "Organizaciones clave en el desarrollo e investigación de la IA generativa." }
];

const complexityLevels = [
  { level: 1, name: "Básico", description: "Conceptos fundamentales, fáciles de entender.", color: "var(--level-basic)" },
  { level: 2, name: "Intermedio", description: "Requiere cierto conocimiento previo.", color: "var(--level-intermediate)" },
  { level: 3, name: "Avanzado", description: "Términos técnicos o especializados.", color: "var(--level-advanced)" }
];

