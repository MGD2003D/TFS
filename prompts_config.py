"""
Конфигурация промптов для почтового ассистента.

Содержит промпты для:
- System prompt LLM
- RAG query prompts (с атрибуцией по письму)
- Query enhancement prompts (email-специфичные интенты)
- Промпты адаптивного поиска (decomposition, multihop, complexity)
"""

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Ты ИИ-ассистент для навигации по почтовой переписке.

Твоя задача — помогать пользователю находить информацию в его письмах: \
даты, договорённости, участников переписки, решения, задачи и сроки.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленных писем
2. НЕ придумывай факты — только то, что есть в переписке
3. Всегда указывай источник: дату письма и отправителя
4. Если информации нет — честно скажи об этом
5. Отвечай чётко и по существу"""


# =============================================================================
# RAG QUERY PROMPT
# =============================================================================

def _format_source_label(metadata: dict) -> str:
    """Строит метку источника из метаданных чанка."""
    parts = []
    if metadata.get("email_date"):
        parts.append(metadata["email_date"])
    if metadata.get("email_from"):
        parts.append(f"от {metadata['email_from']}")
    if metadata.get("email_subject"):
        parts.append(f"тема: «{metadata['email_subject']}»")
    if not parts and metadata.get("source"):
        parts.append(metadata["source"])
    return ", ".join(parts) if parts else "письмо"


def build_rag_prompt(context: str, query: str) -> str:
    """
    Промпт для RAG-ответа с атрибуцией по письму.
    context уже содержит шапки писем (Дата/От/Тема) из индексера.
    """
    return f"""Используй фрагменты писем ниже для ответа на вопрос.

=== ПИСЬМА ===
{context}

=== ВОПРОС ===
{query}

=== ИНСТРУКЦИИ ===
1. Отвечай СТРОГО на основе писем выше
2. Указывай источник: дату и отправителя конкретного письма
3. Если несколько писем касаются вопроса — упомяни все релевантные
4. Если есть противоречия между письмами — отметь это явно
5. Если информации нет — скажи об этом прямо
6. НЕ придумывай детали, которых нет в переписке

Твой ответ:"""


def build_referral_prompt(context: str, query: str) -> str:
    """Промпт когда прямого ответа нет, но в письмах есть смежная информация."""
    return f"""Тебе предоставлены фрагменты писем, но прямого ответа на вопрос в них нет.

=== ПИСЬМА ===
{context}

=== ВОПРОС ===
{query}

=== ИНСТРУКЦИЯ ===
Прямого ответа нет. Изучи письма и:
1. Если есть смежная информация (контакты, ссылки на другие документы, \
упоминания участников) — сообщи об этом с указанием источника
2. Если ничего релевантного нет — скажи, что в загруженной переписке \
ответа не найдено

Твой ответ:"""


# =============================================================================
# QUERY ENHANCEMENT PROMPT
# =============================================================================

QUERY_ENHANCEMENT_PROMPT_TEMPLATE = """Ты эксперт по анализу запросов к базе почтовой переписки.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

ТВОЯ ЗАДАЧА:
1. Определить тип запроса (intent)
2. Извлечь ключевые сущности
3. Переформулировать для улучшения поиска по письмам
4. Сгенерировать альтернативные варианты

ТИПЫ ЗАПРОСОВ (intent):
- "search_by_sender"  — ищет письма от конкретного человека/компании
- "search_by_date"    — ищет письма за период или конкретную дату
- "search_by_topic"   — ищет письма по теме, проекту, предмету
- "find_decision"     — ищет договорённости, решения, согласования
- "find_task"         — ищет задачи, поручения, дедлайны
- "find_contact"      — ищет контакты, реквизиты, адреса
- "summarize_thread"  — просит пересказ цепочки переписки
- "small_talk"        — приветствие, благодарность
- "off_topic"         — вопрос не по теме переписки

ВЕРНИ СТРОГО JSON (без дополнительного текста):
{{
    "intent": "...",
    "rewritten_query": "переформулированный запрос с синонимами",
    "alternative_queries": [
        "альтернативный вариант 1",
        "альтернативный вариант 2"
    ],
    "entities": {{
        "key_terms": ["ключевые слова и фразы"],
        "named_entities": ["люди, компании, проекты"],
        "temporal": ["даты и периоды если упомянуты"],
        "numerical": ["числа и суммы если упомянуты"]
    }}
}}

ПРАВИЛА:
1. Сохраняй имена собственные, названия компаний, проектов точно как в запросе
2. Добавляй синонимы (договор/соглашение, встреча/совещание/звонок)
3. Для поиска по отправителю добавляй вариант с email-форматом
4. Для поиска по дате добавляй разные форматы (15 апреля / 15.04 / April 15)
5. Пустые массивы [] если ничего не найдено
6. ТОЛЬКО JSON, ничего лишнего

ПРИМЕРЫ:

Запрос: "письма от Иванова про договор"
{{
    "intent": "search_by_topic",
    "rewritten_query": "Иванов договор соглашение контракт",
    "alternative_queries": [
        "письмо от Иванова договор подписание",
        "Иванов согласование контракт документ"
    ],
    "entities": {{
        "key_terms": ["договор", "соглашение", "контракт"],
        "named_entities": ["Иванов"],
        "temporal": [],
        "numerical": []
    }}
}}

Запрос: "что решили на прошлой неделе по проекту Альфа"
{{
    "intent": "find_decision",
    "rewritten_query": "решение договорённость проект Альфа",
    "alternative_queries": [
        "согласовано утверждено проект Альфа",
        "итоги обсуждения Альфа договорились"
    ],
    "entities": {{
        "key_terms": ["решение", "договорённость", "итоги"],
        "named_entities": ["проект Альфа"],
        "temporal": ["прошлая неделя"],
        "numerical": []
    }}
}}

Запрос: "дедлайн по сдаче отчёта"
{{
    "intent": "find_task",
    "rewritten_query": "дедлайн срок сдача отчёт",
    "alternative_queries": [
        "когда сдать отчёт срок выполнения",
        "дата сдачи отчёта поручение задача"
    ],
    "entities": {{
        "key_terms": ["дедлайн", "срок", "отчёт"],
        "named_entities": [],
        "temporal": [],
        "numerical": []
    }}
}}

Теперь обработай запрос пользователя. Только JSON:"""


def build_query_enhancement_prompt(query: str) -> str:
    return QUERY_ENHANCEMENT_PROMPT_TEMPLATE.format(query=query)


# =============================================================================
# ASPECT EXTRACTION PROMPT (Query Decomposition)
# =============================================================================

ASPECT_EXTRACTION_PROMPT_TEMPLATE = """You are a query analysis expert for an email search system. \
Decide if a query needs decomposition and extract independent searchable aspects.

CRITICAL RULES:
1. ALWAYS include "original" key with the full original query
2. Simple query (1 concept) → Return ONLY {{"original": "query"}} (triggers baseline)
3. Complex query (2+ independent aspects) → Add 1-4 additional aspects
4. Each aspect must be INDEPENDENT and SEARCHABLE
5. Aspects must NOT semantically overlap
6. Max 5 total aspects (including original)

EXAMPLES:

Simple (NO decomposition):
Input: "письма от Петрова"
Output: {{"original": "письма от Петрова"}}

Complex (decomposition needed):
Input: "что обсуждали с Петровым и Сидоровым про бюджет и сроки"
Output:
{{
  "original": "что обсуждали с Петровым и Сидоровым про бюджет и сроки",
  "petrov": "переписка с Петровым",
  "sidorov": "переписка с Сидоровым",
  "budget": "бюджет финансирование",
  "deadlines": "сроки дедлайны даты"
}}

Now analyze:
{query}

Return ONLY JSON dict with "original" key ALWAYS present. Nothing else."""


def build_aspect_extraction_prompt(query: str) -> str:
    return ASPECT_EXTRACTION_PROMPT_TEMPLATE.format(query=query)


# =============================================================================
# COMPLEXITY ANALYSIS PROMPT (Adaptive routing)
# =============================================================================

COMPLEXITY_ANALYSIS_PROMPT_TEMPLATE = """Analyze the complexity of this email search query and return JSON.

Query: "{query}"

Determine:
1. Is this a multi-hop query? (requires chaining: "кто написал Петрову после того как..." → find email → find reply)
2. Does it have multiple independent aspects? (sender AND topic AND date range)
3. Is it simple? (single lookup)

Return JSON:
{{
    "complexity": "simple | multi_aspect | multi_hop",
    "hops": [
        {{"query": "first lookup", "extract": "what to get from result"}},
        {{"query": "second lookup using {{prev}}", "extract": "final answer"}}
    ],
    "reasoning": "brief explanation"
}}

Only include "hops" for multi_hop complexity. Return ONLY JSON."""


def build_complexity_analysis_prompt(query: str) -> str:
    return COMPLEXITY_ANALYSIS_PROMPT_TEMPLATE.format(query=query)


# =============================================================================
# LETTER COMPOSITION PROMPTS
# =============================================================================

def build_topic_extraction_prompt(letter: str) -> str:
    return f"""Из обращения выдели суть в 2-3 предложениях: тема, конкретный вопрос, адрес объекта если есть.
Только суть, без лишних слов.

{letter[:3000]}"""


def build_letter_composition_prompt(incoming: str, examples: str, legal_context: str) -> str:
    return f"""Составь официальный ответ от Жилищного комитета Санкт-Петербурга на обращение.

=== ВХОДЯЩЕЕ ОБРАЩЕНИЕ ===
{incoming[:4000]}

=== ПОХОЖИЕ ОТВЕТЫ (стиль и структура) ===
{examples}

=== НОРМАТИВНАЯ БАЗА ===
{legal_context}

=== ИНСТРУКЦИИ ===
1. Начни с "Рассмотрев Ваше [обращение/письмо/электронное обращение]..."
2. Используй заглушки [ДАТА] и [ИСХОДЯЩИЙ №] там где нужны реквизиты
3. Цитируй статьи законов из нормативной базы (ЖК РФ, законы СПб)
4. Следуй официально-деловому стилю и структуре похожих ответов
5. НЕ придумывай факты, номера НПА, даты — только из нормативной базы
6. Если нормативная база не содержит нужной информации — укажи это

Черновик ответа:"""


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

ENABLE_QUERY_ENHANCEMENT = True
MIN_RELEVANCE_SCORE = 0.35
DEFAULT_TOP_K = 8
