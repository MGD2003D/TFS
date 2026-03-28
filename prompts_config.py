"""
Конфигурация промптов для RAG системы.

Содержит промпты для:
- System prompt LLM
- RAG query prompts
- Query enhancement prompts
"""

# =============================================================================
# SYSTEM PROMPT для LLM клиента
# =============================================================================

SYSTEM_PROMPT = """Ты ИИ-ассистент администрации Невского района Санкт-Петербурга.

Твоя задача — помогать жителям Санкт-Петербурга находить информацию по вопросам городского управления, жилищным вопросам, социальным услугам и другим темам, связанным с деятельностью администрации. Если вопрос касается конкретно Невского района — учитывай это в ответе.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. СТРОГО используй ТОЛЬКО информацию из предоставленных документов
2. НЕ ПРИДУМЫВАЙ факты, цифры или детали — только то, что есть в документах
3. Если информации недостаточно или её нет — честно скажи об этом
4. Отвечай чётко, структурированно и по существу
5. Если вопрос выходит за рамки предоставленной информации — укажи это
6. Отвечай с учётом специфики Санкт-Петербурга; если вопрос касается Невского района — явно отметь это

Твоя цель — предоставить точную и полезную информацию жителям от лица администрации Невского района Санкт-Петербурга."""


# =============================================================================
# RAG QUERY PROMPT (используется в _build_rag_prompt)
# =============================================================================

def build_referral_prompt(context: str, query: str) -> str:
    """
    Промпт для случая, когда прямого ответа нет, но в тексте могут быть
    контакты/адреса/организации — куда пользователь может обратиться.
    """
    return f"""Тебе предоставлены фрагменты документов, связанных с вопросом, но прямого ответа не содержащих.

=== ФРАГМЕНТЫ ДОКУМЕНТОВ ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

=== ИНСТРУКЦИЯ ===
Прямого ответа на вопрос в документах не найдено. Изучи фрагменты и:
1. Если в тексте есть информация о том, куда обратиться за ответом (организации, адреса, телефоны, сайты, часы работы) — извлеки и сообщи пользователю.
2. Если такой информации тоже нет — скажи, что ответа найти не удалось, и порекомендуй обратиться в администрацию Невского района напрямую.
НЕ придумывай контакты и адреса — только то, что есть в тексте.

Твой ответ:"""


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

QUERY_ENHANCEMENT_PROMPT_TEMPLATE = """Ты эксперт по анализу и улучшению поисковых запросов для базы знаний администрации Невского района Санкт-Петербурга (жилищные вопросы, социальные услуги, городское управление).

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
- "small_talk" - приветствие, благодарность, светская беседа (привет, спасибо, как дела)
- "inappropriate" - грубость, мат, оскорбления
- "off_topic" - вопрос не по теме деятельности администрации

КРИТИЧЕСКИ ВАЖНО — ЖИЛИЩНАЯ ТЕРМИНОЛОГИЯ (НЕ ПУТАТЬ):
- "учётная норма" / "норма учёта" — минимальный метраж на человека, НИЖЕ которого гражданина ставят на жилищный учёт (очередь). СПб: 9 кв.м в отдельных квартирах, 15 кв.м в коммунальных.
- "норма предоставления" — сколько кв.м ДАЮТ при предоставлении жилья (расселении). СПб: 18 кв.м на человека (семья 2+), 33 кв.м одинокому.
- Запросы про "встать на учёт", "встать в очередь", "постановка на учёт", "нуждающийся в улучшении жилищных условий" → используй термины "учётная норма", "норма учёта площади".
- Запросы про "сколько дадут жильё", "при расселении", "при предоставлении" → используй термин "норма предоставления".

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

Запрос: "какой должен быть метраж чтобы встать на очередь"
{{
    "intent": "factual",
    "rewritten_query": "учётная норма площади жилого помещения постановка на жилищный учёт нуждающийся",
    "alternative_queries": [
        "учётная норма 9 квадратных метров жилищный учёт нуждающийся в жилых помещениях Санкт-Петербург",
        "закон Санкт-Петербурга статья 3 учётная норма площади постановка на учёт нуждающийся"
    ],
    "entities": {{
        "key_terms": ["учётная норма", "постановка на жилищный учёт", "нуждающийся в жилых помещениях"],
        "named_entities": ["Санкт-Петербург"],
        "temporal": [],
        "numerical": ["9", "15"]
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

ASPECT_EXTRACTION_PROMPT_TEMPLATE = """You are a query analysis expert. Your task is to decide if a query needs decomposition and extract independent searchable aspects.

CRITICAL RULES:
1. ALWAYS include "original" key with the full original query
2. Simple query (1 concept) → Return ONLY {{"original": "query"}} (triggers baseline)
3. Complex query (2+ concepts) → Add 1-4 additional aspects (decomposition mode)
4. Each aspect should be INDEPENDENT and SEARCHABLE
5. Aspects should NOT semantically overlap
6. Max 5 total aspects (including original)

EXAMPLES:

Simple query (NO decomposition needed):
Input: "что такое фотосинтез"
Output:
{{
  "original": "что такое фотосинтез"
}}
→ Only 1 aspect → System will use BASELINE retrieval

Complex query (decomposition needed):
Input: "Сравни преимущества и риски инвестиций в облигации и акции"
Output:
{{
  "original": "Сравни преимущества и риски инвестиций в облигации и акции",
  "bonds": "преимущества и риски облигаций",
  "stocks": "преимущества и риски акций",
  "comparison": "сравнение облигации vs акции инвестиции"
}}
→ 4 aspects → System will use DECOMPOSITION with weighted fusion

Multi-aspect query:
Input: "Retirement investment with low risk and high liquidity"
Output:
{{
  "original": "Retirement investment with low risk and high liquidity",
  "goal": "retirement investment strategy",
  "risk": "low risk portfolio",
  "liquidity": "high liquidity assets"
}}

Multi-hop query:
Input: "Who is the spouse of the director of Inception?"
Output:
{{
  "original": "Who is the spouse of the director of Inception?",
  "movie_director": "director of Inception",
  "director_spouse": "Christopher Nolan spouse"
}}

Now analyze:
{query}

Return ONLY JSON dict with "original" key ALWAYS present. Nothing else."""


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
# НАСТРОЙКИ
# =============================================================================

# Включить/выключить query enhancement для тестирования
ENABLE_QUERY_ENHANCEMENT = True  # Можно отключить для базового RAG тестирования

# Минимальная релевантность для фильтрации результатов
MIN_RELEVANCE_SCORE = 0.35

# Количество документов для retrieval
DEFAULT_TOP_K = 8
