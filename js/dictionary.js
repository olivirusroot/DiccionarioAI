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
    id: "gemas",
    term: "Gemas (Gems)",
    shortDefinition: "Modelos de lenguaje especializados de Google para tareas específicas (ecosistema Gemini).",
    fullDefinition: "En el contexto de Google, modelos de lenguaje especializados diseñados para tareas específicas dentro del ecosistema Gemini.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["gemini", "google", "fine-tuning"],
    tags: ["google", "modelo", "especialización"]
  },
  {
    id: "embedding-contextualizado",
    term: "Incorporación de Lenguaje Contextualizado",
    shortDefinition: "Embedding que comprende palabras considerando su contexto específico.",
    fullDefinition: "Tipo de embedding que comprende palabras y frases considerando su contexto específico, captando matices semánticos complejos.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["embedding", "bert", "nlp"],
    tags: ["nlp", "representación", "semántica"]
  },
  {
    id: "jailbreaking",
    term: "Jailbreaking",
    shortDefinition: "Técnicas para eludir las salvaguardas éticas y restricciones de seguridad en modelos IA.",
    fullDefinition: "Técnicas utilizadas para eludir las salvaguardas éticas y restricciones de seguridad implementadas en modelos de IA generativa.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["seguridad", "alineacion-ia", "prompt-engineering"],
    tags: ["seguridad", "riesgo"]
  },
  {
    id: "tensor-cores",
    term: "NVIDIA Tensor Cores",
    shortDefinition: "Unidades de procesamiento en GPUs NVIDIA para acelerar operaciones de redes neuronales.",
    fullDefinition: "Unidades de procesamiento especializadas en GPUs de NVIDIA diseñadas específicamente para acelerar operaciones de álgebra matricial utilizadas en redes neuronales.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["gpu", "aprendizaje-profundo", "nvidia"],
    tags: ["hardware", "aceleración"]
  },
  {
    id: "chatear",
    term: "Chatear",
    shortDefinition: "Diálogo de ida y vuelta con un sistema de IA, donde la interacción previa es contexto.",
    fullDefinition: "Contenido de un diálogo de ida y vuelta con un sistema de IA, donde la interacción anterior se convierte en contexto para las partes posteriores de la conversación.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["chatgpt", "memoria-de-contexto"],
    tags: ["interacción", "conversacional"]
  },
  {
    id: "evals",
    term: "Evals",
    shortDefinition: "Abreviatura de evaluaciones, especialmente para medir rendimiento de LLMs.",
    fullDefinition: "Abreviatura de evaluaciones, especialmente en el contexto de medir el rendimiento y la calidad de los modelos de lenguaje grandes.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["evaluacion-automatica", "perplexity"],
    tags: ["evaluación", "jerga"]
  },
  {
    id: "midjourney",
    term: "Midjourney",
    shortDefinition: "Herramienta de IA generativa especializada en crear imágenes artísticas desde texto.",
    fullDefinition: "Herramienta de IA generativa especializada en crear imágenes artísticas de alta calidad a partir de descripciones textuales.",
    category: "aplicaciones",
    complexity: 1,
    relatedTerms: ["dall-e", "stable-diffusion", "texto-a-imagen"],
    tags: ["imágenes", "arte", "aplicaciones"]
  },
  {
    id: "anthropic",
    term: "Anthropic",
    shortDefinition: "Empresa de investigación en IA que desarrolló Claude, enfocada en seguridad.",
    fullDefinition: "Empresa de investigación en IA que desarrolló Claude, enfocada en la seguridad y alineación de sistemas de IA avanzados.",
    category: "empresas",
    complexity: 2,
    relatedTerms: ["claude", "seguridad", "alineacion-ia"],
    tags: ["empresa", "investigación", "seguridad"]
  },
  {
    id: "bard",
    term: "Bard",
    shortDefinition: "Asistente de IA conversacional de Google, predecesor de Gemini.",
    fullDefinition: "Asistente de IA conversacional desarrollado por Google, predecesor de Gemini, diseñado para interactuar con usuarios y proporcionar información.",
    category: "aplicaciones",
    complexity: 1,
    relatedTerms: ["google", "gemini", "chatgpt"],
    tags: ["google", "chatbot", "histórico"]
  },
  {
    id: "copilot",
    term: "Copilot",
    shortDefinition: "Asistente de IA de Microsoft/GitHub que ayuda a escribir código.",
    fullDefinition: "Asistente de IA desarrollado por Microsoft y GitHub que ayuda a los programadores a escribir código sugiriendo líneas o funciones completas.",
    category: "aplicaciones",
    complexity: 1,
    relatedTerms: ["microsoft", "github", "codigo"],
    tags: ["microsoft", "desarrollo", "código"]
  },
  {
    id: "hugging-face",
    term: "Hugging Face",
    shortDefinition: "Plataforma con herramientas, modelos y datasets para IA, especialmente NLP.",
    fullDefinition: "Plataforma que proporciona herramientas, modelos y datasets para aplicaciones de IA, especialmente en procesamiento de lenguaje natural.",
    category: "empresas",
    complexity: 2,
    relatedTerms: ["nlp", "modelo-base", "código-abierto"],
    tags: ["plataforma", "comunidad", "nlp"]
  },
  {
    id: "whisper",
    term: "Whisper",
    shortDefinition: "Modelo de reconocimiento de voz de OpenAI para transcribir y traducir audio.",
    fullDefinition: "Modelo de reconocimiento de voz desarrollado por OpenAI que puede transcribir y traducir audio en múltiples idiomas con alta precisión.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["openai", "reconocimiento-voz", "audio"],
    tags: ["openai", "audio", "transcripción"]
  },
  {
    id: "sora",
    term: "Sora",
    shortDefinition: "Modelo de IA generativa de OpenAI capaz de crear videos realistas desde texto.",
    fullDefinition: "Modelo de IA generativa de OpenAI capaz de crear videos realistas a partir de descripciones textuales, representando un avance en la generación de contenido visual dinámico.",
    category: "modelos",
    complexity: 3,
    relatedTerms: ["openai", "video", "modelo-multimodal"],
    tags: ["openai", "video", "texto-a-video"]
  },
  {
    id: "llama",
    term: "Llama",
    shortDefinition: "Familia de LLMs de código abierto desarrollados por Meta.",
    fullDefinition: "Familia de modelos de lenguaje grandes de código abierto desarrollados por Meta (anteriormente Facebook), diseñados para ser más accesibles para investigadores y desarrolladores.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["meta", "llm", "código-abierto"],
    tags: ["meta", "modelo", "código-abierto"]
  },
  {
    id: "mistral",
    term: "Mistral",
    shortDefinition: "Modelo de lenguaje de código abierto eficiente y potente.",
    fullDefinition: "Modelo de lenguaje de código abierto que ofrece capacidades avanzadas con requisitos computacionales más eficientes que otros LLMs de tamaño similar.",
    category: "modelos",
    complexity: 2,
    relatedTerms: ["llm", "código-abierto", "eficiencia"],
    tags: ["modelo", "código-abierto", "eficiencia"]
  },
  {
    id: "imagen-latente",
    term: "Imagen Latente",
    shortDefinition: "Representación comprimida de una imagen en el espacio latente de un modelo.",
    fullDefinition: "Representación comprimida de una imagen en el espacio latente de un modelo generativo, que captura características esenciales de forma matemática.",
    category: "conceptos",
    complexity: 3,
    relatedTerms: ["espacio-latente", "difusion", "vae"],
    tags: ["imágenes", "representación"]
  },
  {
    id: "alineacion-ia",
    term: "Alineación de IA",
    shortDefinition: "Asegurar que los sistemas de IA actúen según las intenciones y valores humanos.",
    fullDefinition: "Proceso de asegurar que los sistemas de IA actúen de acuerdo con las intenciones y valores humanos, minimizando comportamientos no deseados.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["rlhf", "seguridad", "etica"],
    tags: ["ética", "seguridad", "control"]
  },
  {
    id: "sesgo-algoritmico",
    term: "Sesgo Algorítmico",
    shortDefinition: "Tendencia sistemática de un modelo a producir resultados injustos.",
    fullDefinition: "Tendencia sistemática de un modelo de IA a producir resultados que favorecen o perjudican injustamente a ciertos grupos, reflejando sesgos presentes en los datos de entrenamiento.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["etica", "datos", "entrenamiento"],
    tags: ["ética", "riesgo", "justicia"]
  },
  {
    id: "interpretabilidad",
    term: "Interpretabilidad",
    shortDefinition: "Grado en que se puede entender y explicar el funcionamiento interno de un modelo IA.",
    fullDefinition: "Grado en que se puede entender y explicar el funcionamiento interno y las decisiones de un modelo de IA, crucial para aplicaciones críticas y confianza en los sistemas.",
    category: "conceptos",
    complexity: 2,
    relatedTerms: ["explicabilidad", "confianza", "etica"],
    tags: ["ética", "transparencia"]
  },
  {
    id: "prompt",
    term: "Prompt",
    shortDefinition: "Instrucción o entrada de texto proporcionada a un modelo generativo para guiar su salida.",
    fullDefinition: "La instrucción o entrada de texto que se proporciona a un modelo de lenguaje grande (LLM) u otro modelo generativo para condicionar su respuesta o salida. Diseñar prompts efectivos es clave en la ingeniería de prompts.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["prompt-engineering", "in-context-learning"],
    tags: ["interacción", "entrada"]
  },
  {
    id: "token",
    term: "Token",
    shortDefinition: "Unidad fundamental de texto (palabra, subpalabra, carácter) que procesa un modelo.",
    fullDefinition: "La unidad atómica en la que un modelo de lenguaje divide el texto. Un token puede ser una palabra, una parte de una palabra (subpalabra) o incluso un solo carácter, dependiendo del método de tokenización utilizado.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["tokenizacion", "ventana-de-contexto"],
    tags: ["nlp", "procesamiento", "unidad"]
  },
  {
    id: "regularizacion",
    term: "Regularización",
    shortDefinition: "Técnicas para prevenir el sobreajuste en modelos de aprendizaje automático.",
    fullDefinition: "Conjunto de técnicas utilizadas durante el entrenamiento de modelos de aprendizaje automático para prevenir el sobreajuste (overfitting), mejorando así la capacidad del modelo para generalizar a datos no vistos. Ejemplos incluyen L1/L2 y weight decay.",
    category: "tecnicas",
    complexity: 2,
    relatedTerms: ["overfitting", "weight-decay", "entrenamiento"],
    tags: ["entrenamiento", "optimización"]
  },
  {
    id: "top-k",
    term: "Top-k Sampling",
    shortDefinition: "Estrategia de muestreo que considera solo los k tokens más probables.",
    fullDefinition: "Una estrategia de muestreo utilizada en la generación de texto donde, en cada paso, el modelo considera solo los 'k' tokens más probables y redistribuye la probabilidad entre ellos antes de seleccionar el siguiente token.",
    category: "tecnicas",
    complexity: 3,
    relatedTerms: ["sampling", "temperatura", "generacion-condicionada"],
    tags: ["generación", "probabilidad", "muestreo"]
  },
  {
    id: "gpu",
    term: "GPU (Graphics Processing Unit)",
    shortDefinition: "Procesador especializado crucial para acelerar el entrenamiento de modelos de IA.",
    fullDefinition: "Unidad de Procesamiento Gráfico. Procesador especializado diseñado originalmente para gráficos por computadora, pero que es fundamental para acelerar las operaciones matemáticas paralelas requeridas en el entrenamiento y la inferencia de modelos de aprendizaje profundo.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["tensor-cores", "aprendizaje-profundo", "nvidia"],
    tags: ["hardware", "aceleración"]
  },
  {
    id: "nlp",
    term: "NLP (Natural Language Processing)",
    shortDefinition: "Campo de la IA enfocado en la interacción entre computadoras y lenguaje humano.",
    fullDefinition: "Procesamiento del Lenguaje Natural. Campo de la inteligencia artificial y la lingüística computacional que se enfoca en la interacción entre las computadoras y el lenguaje humano, incluyendo la comprensión y generación de texto.",
    category: "conceptos",
    complexity: 1,
    relatedTerms: ["llm", "bert", "tokenizacion"],
    tags: ["ia", "lenguaje"]
  },
  {
    id: "openai",
    term: "OpenAI",
    shortDefinition: "Laboratorio de investigación en IA que desarrolló GPT, DALL-E, ChatGPT, Sora y Whisper.",
    fullDefinition: "Organización de investigación y desarrollo en inteligencia artificial, conocida por crear modelos influyentes como la serie GPT, DALL-E, ChatGPT, Sora y Whisper.",
    category: "empresas",
    complexity: 1,
    relatedTerms: ["gpt", "chatgpt", "dall-e", "sora", "whisper"],
    tags: ["empresa", "investigación"]
  },
  {
    id: "google",
    term: "Google",
    shortDefinition: "Empresa tecnológica pionera en IA, desarrolladora de Transformer, BERT, Gemini y Bard.",
    fullDefinition: "Gigante tecnológico que ha realizado contribuciones fundamentales a la IA, incluyendo la arquitectura Transformer, modelos como BERT y Gemini (sucesor de Bard), y la plataforma Google Cloud AI.",
    category: "empresas",
    complexity: 1,
    relatedTerms: ["arquitectura-transformer", "bert", "gemini", "bard", "tensor-cores"],
    tags: ["empresa", "investigación"]
  },
  {
    id: "meta",
    term: "Meta",
    shortDefinition: "Empresa tecnológica (antes Facebook) que desarrolló la familia de modelos Llama.",
    fullDefinition: "Empresa tecnológica anteriormente conocida como Facebook, que ha invertido significativamente en IA y ha lanzado modelos de lenguaje grandes de código abierto como la familia Llama.",
    category: "empresas",
    complexity: 1,
    relatedTerms: ["llama", "llm", "código-abierto"],
    tags: ["empresa", "investigación"]
  },
  {
    id: "microsoft",
    term: "Microsoft",
    shortDefinition: "Empresa tecnológica con fuerte inversión en IA, asociada con OpenAI y desarrolladora de Copilot.",
    fullDefinition: "Empresa tecnológica que integra fuertemente la IA en sus productos y servicios (Azure AI, Microsoft 365 Copilot), y tiene una asociación estratégica con OpenAI.",
    category: "empresas",
    complexity: 1,
    relatedTerms: ["copilot", "openai", "azure"],
    tags: ["empresa", "aplicaciones"]
  },
  {
    id: "nvidia",
    term: "NVIDIA",
    shortDefinition: "Empresa líder en GPUs y hardware especializado para aceleración de IA.",
    fullDefinition: "Empresa líder en el diseño y fabricación de Unidades de Procesamiento Gráfico (GPUs) y hardware especializado (como Tensor Cores) que son cruciales para el entrenamiento y la inferencia de modelos de IA a gran escala.",
    category: "empresas",
    complexity: 1,
    relatedTerms: ["gpu", "tensor-cores", "hardware"],
    tags: ["empresa", "hardware"]
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

