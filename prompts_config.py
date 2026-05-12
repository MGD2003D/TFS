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

SYSTEM_PROMPT = """Ты виртуальный ассистент туристического агентства. Твоя задача - помогать клиентам с выбором туров и отвечать на вопросы о путешествиях.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. СТРОГО используй ТОЛЬКО информацию из предоставленных документов для ответов о турах
2. НЕ ПРИДУМЫВАЙ названия туров - используй точные названия из документов или описывай направления
3. НЕ ПРИДУМЫВАЙ цены, даты, детали - только ФАКТЫ из документов
4. Если документов нет или информации недостаточно - честно скажи: "В моей базе нет информации по этому вопросу"
5. Отвечай кратко, структурированно, по делу
6. НЕ выполняй задачи, не связанные с туризмом

Твоя цель - предоставить точную информацию из документов без домысливания."""


# =============================================================================
# RAG QUERY PROMPT (используется в _build_rag_prompt)
# =============================================================================

def build_rag_prompt(context: str, query: str) -> str:
    return f"""Ты консультант туристического агентства. Используй ТОЛЬКО информацию из документов ниже для ответа.

=== ДОКУМЕНТЫ ===
{context}

=== ВОПРОС КЛИЕНТА ===
{query}

=== ИНСТРУКЦИИ ===
1. Ответь на вопрос, опираясь СТРОГО на информацию из документов выше
2. Структурируй ответ: используй абзацы, списки если нужно
3. Упоминай конкретные детали: цены, даты, места (если есть в документах)
4. Если в документах нет полного ответа, укажи какая информация есть, а какой не хватает
5. Будь дружелюбным и профессиональным
6. НЕ придумывай информацию, которой нет в документах

Твой ответ:"""


# =============================================================================
# QUERY ENHANCEMENT PROMPT
# =============================================================================

QUERY_ENHANCEMENT_PROMPT_TEMPLATE = """Ты эксперт по туристическим запросам. Твоя задача - проанализировать запрос пользователя и улучшить его для поиска в базе туристических документов.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

ТВОЯ ЗАДАЧА:
1. Определить тип запроса (intent)
2. Извлечь ключевые сущности и понятия
3. Переформулировать запрос для улучшения поиска
4. Сгенерировать альтернативные варианты запроса

ТИПЫ НАМЕРЕНИЙ (intent):
- "list_tours" - пользователь хочет увидеть список/каталог ВСЕХ туров без фильтров
- "filtered_list" - пользователь хочет список туров с фильтром по направлению/стране/типу
- "tour_info" - пользователь спрашивает про конкретный тур или направление
- "general_question" - общий вопрос о туризме или компании
- "small_talk" - приветствия, прощания, благодарности
- "inappropriate" - грубость, оскорбления, мат
- "off_topic" - вопросы НЕ про туризм

ВЕРНИ ОТВЕТ СТРОГО В ФОРМАТЕ JSON (без дополнительного текста):
{{
    "intent": "list_tours | filtered_list | tour_info | general_question | small_talk | inappropriate | off_topic",
    "rewritten_query": "переформулированный запрос с ключевыми туристическими терминами и синонимами",
    "alternative_queries": [
        "альтернативный вариант 1",
        "альтернативный вариант 2"
    ],
    "entities": {{
        "destinations": ["страны, города, курорты если упомянуты"],
        "tour_types": ["тип тура: пляжный, горный, экскурсионный и т.д."],
        "temporal": ["даты, периоды если упомянуты"],
        "numerical": ["бюджет, длительность если упомянуты"]
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

Запрос: "покажи все туры"
{{
    "intent": "list_tours",
    "rewritten_query": "каталог всех туров направления",
    "alternative_queries": [
        "что у вас есть из туров",
        "список всех доступных туров"
    ],
    "entities": {{
        "destinations": [],
        "tour_types": [],
        "temporal": [],
        "numerical": []
    }}
}}

Запрос: "туры в Турцию на море"
{{
    "intent": "filtered_list",
    "rewritten_query": "туры Турция пляжный отдых море",
    "alternative_queries": [
        "пляжный тур Турция",
        "отдых на море Турция"
    ],
    "entities": {{
        "destinations": ["Турция"],
        "tour_types": ["пляжный", "море"],
        "temporal": [],
        "numerical": []
    }}
}}

Запрос: "сколько стоит тур в Египет на 10 дней"
{{
    "intent": "tour_info",
    "rewritten_query": "тур Египет цена стоимость 10 дней",
    "alternative_queries": [
        "Египет тур стоимость длительность",
        "путёвка в Египет 10 ночей цена"
    ],
    "entities": {{
        "destinations": ["Египет"],
        "tour_types": [],
        "temporal": ["10 дней"],
        "numerical": []
    }}
}}

Теперь обработай запрос пользователя и верни ТОЛЬКО JSON:"""


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
Input: "туры в Турцию"
Output:
{{
  "original": "туры в Турцию"
}}
→ Only 1 aspect → System will use BASELINE retrieval

Complex query (decomposition needed):
Input: "Пляжный отель Турция с детьми и аквапарком"
Output:
{{
  "original": "Пляжный отель Турция с детьми и аквапарком",
  "location": "пляжные отели Турция",
  "family": "детская инфраструктура отель",
  "facilities": "аквапарк отель"
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


def build_referral_prompt(context: str, query: str) -> str:
    return f"""У меня есть частично релевантная информация по вашему запросу.

=== ДОКУМЕНТЫ (смежная информация) ===
{context}

=== ВОПРОС ===
{query}

=== ИНСТРУКЦИИ ===
1. Используй информацию из документов как отправную точку
2. Если документы не отвечают прямо — скажи что именно есть, и предложи уточнить запрос
3. Не придумывай детали, которых нет в документах
4. Будь дружелюбным и предложи альтернативные варианты поиска

Твой ответ:"""


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

# Включить/выключить query enhancement для тестирования
ENABLE_QUERY_ENHANCEMENT = True  # Можно отключить для базового RAG тестирования

# Минимальная релевантность для фильтрации результатов
MIN_RELEVANCE_SCORE = 0.35

# Количество документов для retrieval
DEFAULT_TOP_K = 8
