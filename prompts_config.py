"""
Конфигурация промптов для RAG системы.

Содержит промпты для:
- System prompt LLM
- RAG query prompts
- Query enhancement prompts

Для RAGAS тестирования используются общие промпты.
Туристические промпты оставлены закомментированными для возможности переключения.
"""

# =============================================================================
# SYSTEM PROMPT для LLM клиента
# =============================================================================

# --- ТУРИСТИЧЕСКОЕ АГЕНТСТВО (закомментировано) ---
# SYSTEM_PROMPT = """Ты виртуальный ассистент туристического агентства. Твоя задача - помогать клиентам с выбором туров и отвечать на вопросы о путешествиях.
#
# КРИТИЧЕСКИЕ ПРАВИЛА:
# 1. СТРОГО используй ТОЛЬКО информацию из предоставленных документов для ответов о турах
# 2. НЕ ПРИДУМЫВАЙ названия туров - используй точные названия из документов или описывай направления
# 3. НЕ ПРИДУМЫВАЙ цены, даты, детали - только ФАКТЫ из документов
# 4. Если документов нет или информации недостаточно - честно скажи: "В моей базе нет информации по этому вопросу"
# 5. Отвечай кратко, структурированно, по делу
# 6. НЕ выполняй задачи, не связанные с туризмом
#
# Твоя цель - предоставить точную информацию из документов без домысливания."""

# --- ОБЩИЙ RAG АССИСТЕНТ (для RAGAS тестирования) ---
SYSTEM_PROMPT = """Ты полезный AI ассистент, который отвечает на вопросы пользователей на основе предоставленной информации из базы знаний.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. СТРОГО используй ТОЛЬКО информацию из предоставленных документов
2. НЕ ПРИДУМЫВАЙ факты, цифры или детали - только то, что есть в документах
3. Если информации недостаточно или её нет - честно скажи об этом
4. Отвечай чётко, структурированно и по существу
5. Если вопрос выходит за рамки предоставленной информации - укажи это

Твоя цель - предоставить точную и полезную информацию на основе документов."""


# =============================================================================
# RAG QUERY PROMPT (используется в _build_rag_prompt)
# =============================================================================

# --- ТУРИСТИЧЕСКОЕ АГЕНТСТВО (закомментировано) ---
# def build_rag_prompt(context: str, query: str) -> str:
#     return f"""Ты консультант туристического агентства. Используй ТОЛЬКО информацию из документов ниже для ответа.
#
# === ДОКУМЕНТЫ ===
# {context}
#
# === ВОПРОС КЛИЕНТА ===
# {query}
#
# === ИНСТРУКЦИИ ===
# 1. Ответь на вопрос, опираясь СТРОГО на информацию из документов выше
# 2. Структурируй ответ: используй абзацы, списки если нужно
# 3. Упоминай конкретные детали: цены, даты, места (если есть в документах)
# 4. Если в документах нет полного ответа, укажи какая информация есть, а какой не хватает
# 5. Будь дружелюбным и профессиональным
# 6. НЕ придумывай информацию, которой нет в документах
#
# Твой ответ:"""

# --- ОБЩИЙ RAG (для RAGAS тестирования) ---
def build_rag_prompt(context: str, query: str) -> str:
    """
    Строит промпт для RAG запроса с контекстом из документов.

    Args:
        context: Релевантная информация из документов
        query: Вопрос пользователя

    Returns:
        Готовый промпт для LLM
    """
    return f"""Используй информацию из документов ниже для ответа на вопрос пользователя.

=== ДОКУМЕНТЫ ===
{context}

=== ВОПРОС ===
{query}

=== ИНСТРУКЦИИ ===
1. Ответь на вопрос, используя СТРОГО информацию из документов выше
2. Если ответ требует нескольких пунктов - структурируй его (списки, абзацы)
3. Приводи конкретные факты и детали из документов
4. Если информации недостаточно - укажи, что известно, а что нет
5. Будь точным и лаконичным
6. НЕ придумывай информацию, которой нет в документах

Твой ответ:"""


# =============================================================================
# QUERY ENHANCEMENT PROMPT
# =============================================================================

# --- ТУРИСТИЧЕСКОЕ АГЕНТСТВО (закомментировано) ---
# QUERY_ENHANCEMENT_PROMPT_TEMPLATE = """Ты эксперт по туристическим запросам. Твоя задача - проанализировать запрос пользователя и улучшить его для поиска в базе туристических документов.
#
# ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"
#
# ТВОЯ ЗАДАЧА:
# 1. Определить НАМЕРЕНИЕ пользователя (intent)
# 2. Извлечь ключевые сущности из запроса
# 3. Переформулировать запрос для улучшения поиска
# 4. Сгенерировать альтернативные варианты запроса
#
# ТИПЫ НАМЕРЕНИЙ (intent):
# - "list_tours" - пользователь хочет увидеть список/каталог ВСЕХ туров без фильтров
# - "filtered_list" - пользователь хочет список туров с фильтром по направлению/стране/типу
# - "tour_info" - пользователь спрашивает про конкретный тур/направление
# - "general_question" - общий вопрос о туризме или компании
# - "small_talk" - приветствия, прощания, благодарности
# - "inappropriate" - грубость, оскорбления, мат
# - "off_topic" - вопросы НЕ про туризм
# ...
# [остальной промпт]
# """

# --- ОБЩИЙ RAG (для RAGAS тестирования) ---
QUERY_ENHANCEMENT_PROMPT_TEMPLATE = """Ты эксперт по анализу и улучшению поисковых запросов. Твоя задача - проанализировать запрос пользователя и улучшить его для поиска в базе знаний.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

ТВОЯ ЗАДАЧА:
1. Определить тип запроса (intent)
2. Извлечь ключевые сущности и понятия
3. Переформулировать запрос для улучшения поиска
4. Сгенерировать альтернативные варианты запроса

ТИПЫ ЗАПРОСОВ (intent):
- "factual" - фактический вопрос, требующий конкретной информации
- "definition" - запрос определения или объяснения термина
- "comparison" - сравнение нескольких объектов/понятий
- "process" - как что-то работает, процесс или инструкция
- "general" - общий вопрос

ВЕРНИ ОТВЕТ СТРОГО В ФОРМАТЕ JSON (без дополнительного текста):
{{
    "intent": "factual | definition | comparison | process | general",
    "rewritten_query": "переформулированный запрос с ключевыми терминами и синонимами",
    "alternative_queries": [
        "альтернативный вариант 1",
        "альтернативный вариант 2"
    ],
    "entities": {{
        "key_terms": ["список ключевых терминов и понятий"],
        "named_entities": ["имена, места, организации если упомянуты"],
        "temporal": ["даты, периоды если упомянуты"],
        "numerical": ["числа, количества если упомянуты"]
    }}
}}

CRITICAL RULES:
1. **PRESERVE SPECIALIZED TERMS**:
   - Keep ALL-CAPS acronyms unchanged (e.g., DNA, API, ROE)
   - Keep domain-specific technical terms as-is
   - Keep brand/product names with exact spelling
   - Keep numbered standards/codes unchanged (e.g., 401k, COVID-19)
   - DO NOT expand abbreviations unless you're certain it helps search

2. **MAINTAIN SPECIFICITY**:
   - Don't replace specific terms with vague generic ones
   - Add related concepts, don't substitute
   - Keep the original precision of the query

3. **EXPAND, DON'T REPLACE**:
   - Include original terms AND synonyms/related concepts
   - Pattern: "original + synonym + related" NOT "synonym only"

4. **USEFUL VARIANTS ONLY**:
   - Generate variants that offer DIFFERENT search angles BUT SAME INTENT
   - Use different keywords, synonyms, or framing for the SAME underlying question
   - Each variant should help find the SAME information through different wording
   - Skip variants that are just minor rephrasings
   - NEVER generate variants with opposite or contradictory meaning

5. **JSON FORMAT**:
   - Empty arrays [] when no entities found
   - NO text outside JSON

ПРИМЕРЫ:

Запрос: "What is photosynthesis?"
{{
    "intent": "definition",
    "rewritten_query": "photosynthesis process plant cells chlorophyll light energy",
    "alternative_queries": [
        "how does photosynthesis work in plants",
        "photosynthesis definition biology"
    ],
    "entities": {{
        "key_terms": ["photosynthesis", "plant biology", "cellular process"],
        "named_entities": [],
        "temporal": [],
        "numerical": []
    }}
}}

Запрос: "sociologists define ethnicity as a system for classifying people"
{{
    "intent": "definition",
    "rewritten_query": "ethnicity definition sociology classification system social groups",
    "alternative_queries": [
        "how do sociologists define ethnicity",
        "sociological concept of ethnicity classification"
    ],
    "entities": {{
        "key_terms": ["ethnicity", "sociology", "classification", "social groups"],
        "named_entities": [],
        "temporal": [],
        "numerical": []
    }}
}}

Запрос: "how long can you freeze salmon for"
{{
    "intent": "factual",
    "rewritten_query": "salmon freezing duration storage time frozen fish",
    "alternative_queries": [
        "maximum time to freeze salmon",
        "how long salmon stays good frozen"
    ],
    "entities": {{
        "key_terms": ["salmon", "freezing", "storage duration", "food preservation"],
        "named_entities": ["salmon"],
        "temporal": [],
        "numerical": []
    }}
}}

Запрос: "What does high operating margin but small positive ROE imply?"
{{
    "intent": "factual",
    "rewritten_query": "high operating margin low ROE financial performance profitability efficiency capital structure",
    "alternative_queries": [
        "operating margin vs ROE relationship company analysis",
        "high operating margin with low return on equity implications"
    ],
    "entities": {{
        "key_terms": ["operating margin", "ROE", "profitability", "financial metrics", "capital efficiency"],
        "named_entities": [],
        "temporal": [],
        "numerical": ["high", "small", "positive"]
    }}
}}

Now process the user's query and return ONLY JSON:"""


def build_query_enhancement_prompt(query: str) -> str:
    """
    Строит промпт для улучшения поискового запроса.

    Args:
        query: Исходный запрос пользователя

    Returns:
        Готовый промпт для LLM
    """
    return QUERY_ENHANCEMENT_PROMPT_TEMPLATE.format(query=query)


# =============================================================================
# ASPECT EXTRACTION PROMPT (для Query Decomposition)
# =============================================================================

ASPECT_EXTRACTION_PROMPT_TEMPLATE = """You are a query analysis expert. Classify the query type and extract aspects if needed.

QUERY TYPES:
- "simple" → Single concept, no decomposition needed
- "parallel" → Multiple INDEPENDENT aspects that can be searched separately and merged
- "sequential" → Multi-hop chain where each step DEPENDS on the previous result

CRITICAL RULES:
1. ALWAYS include "type" and "original" fields
2. Simple query → {{"type": "simple", "original": "query"}}
3. Parallel query → {{"type": "parallel", "original": "query", "aspects": {{...}}}}
4. Sequential query → {{"type": "sequential", "original": "query", "hops": [...]}}
5. For parallel: aspects must be INDEPENDENT and SEARCHABLE, max 4 aspects
6. For sequential: hops are ORDERED, each hop may use "{{prev}}" placeholder for previous hop's result
7. Keep aspect/hop queries concise and search-optimized

EXAMPLES:

Simple (1 concept):
Input: "туры в Турцию"
Output:
{{
  "type": "simple",
  "original": "туры в Турцию"
}}

Parallel (independent aspects):
Input: "Пляжный отель Турция с детьми и аквапарком"
Output:
{{
  "type": "parallel",
  "original": "Пляжный отель Турция с детьми и аквапарком",
  "aspects": {{
    "location": "пляжные отели Турция",
    "family": "детская инфраструктура отель",
    "facilities": "аквапарк отель"
  }}
}}

Parallel (multiple independent criteria):
Input: "Retirement investment with low risk and high liquidity"
Output:
{{
  "type": "parallel",
  "original": "Retirement investment with low risk and high liquidity",
  "aspects": {{
    "goal": "retirement investment strategy",
    "risk": "low risk portfolio",
    "liquidity": "high liquidity assets"
  }}
}}

Sequential (multi-hop, each step depends on previous):
Input: "Who is the spouse of the director of Inception?"
Output:
{{
  "type": "sequential",
  "original": "Who is the spouse of the director of Inception?",
  "hops": [
    {{"query": "director of Inception", "extract": "director name"}},
    {{"query": "{{prev}} spouse wife husband", "extract": "spouse name"}}
  ]
}}

Sequential (chained lookup):
Input: "Какие экскурсии есть в городе, где находится отель из тура Золотая Анталья?"
Output:
{{
  "type": "sequential",
  "original": "Какие экскурсии есть в городе, где находится отель из тура Золотая Анталья?",
  "hops": [
    {{"query": "тур Золотая Анталья отель город", "extract": "город отеля"}},
    {{"query": "экскурсии {{prev}}", "extract": "список экскурсий"}}
  ]
}}

Sequential (comparative):
Input: "Is the country with the highest GDP in Europe also the most populated?"
Output:
{{
  "type": "sequential",
  "original": "Is the country with the highest GDP in Europe also the most populated?",
  "hops": [
    {{"query": "highest GDP country Europe", "extract": "country name"}},
    {{"query": "most populated country Europe vs {{prev}} population", "extract": "population comparison"}}
  ]
}}

Now analyze:
{query}

Return ONLY JSON. Nothing else."""


def build_aspect_extraction_prompt(query: str) -> str:
    """
    Строит промпт для извлечения аспектов из запроса (query decomposition).

    Args:
        query: Исходный запрос пользователя

    Returns:
        Готовый промпт для LLM
    """
    return ASPECT_EXTRACTION_PROMPT_TEMPLATE.format(query=query)


# =============================================================================
# TRIPLET EXTRACTION PROMPT (для фонового обогащения Knowledge Graph)
# =============================================================================

TRIPLET_EXTRACTION_PROMPT_TEMPLATE = """Extract factual entity relationships from the search results below.

QUERY: {query}

SEARCH RESULTS:
{context}

Your task: find named entity relationships (triplets) explicitly stated in the documents.

Return JSON:
{{
  "triplets": [
    {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0}}
  ]
}}

Rules:
- subject/object: specific named entities (people, places, organizations, products, tours)
- predicate: short relation (located_in, directed_by, part_of, has_feature, spouse_of, includes, costs, departs_from)
- confidence: 0.9 = explicitly stated, 0.75 = clearly implied, skip if uncertain (<0.7)
- Max 5 most important triplets
- Only facts FROM THE DOCUMENTS, not from your knowledge
- If no clear entity relations found, return {{"triplets": []}}

Return ONLY JSON:"""


def build_triplet_extraction_prompt(query: str, context: str) -> str:
    """
    Строит промпт для извлечения триплетов из документов.

    Args:
        query: исходный запрос
        context: текст документов (до 1500 символов)

    Returns:
        Готовый промпт для LLM
    """
    return TRIPLET_EXTRACTION_PROMPT_TEMPLATE.format(query=query, context=context)


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

# Включить/выключить query enhancement для тестирования
ENABLE_QUERY_ENHANCEMENT = True  # Можно отключить для базового RAG тестирования

# Минимальная релевантность для фильтрации результатов
MIN_RELEVANCE_SCORE = 0.35

# Количество документов для retrieval
DEFAULT_TOP_K = 8
