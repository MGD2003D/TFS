import app_state
from typing import List, Dict, Optional
import json


class TourCatalogService:
    """
    Сервис для создания и управления каталогом туров с названиями и описаниями.

    При старте приложения:
    1. Проходит по всем документам
    2. Для каждого генерирует краткое описание через LLM
    3. Сохраняет в памяти для быстрого доступа
    """

    def __init__(self, generate_descriptions: bool = True):
        self.catalog: Dict[str, Dict] = {}
        self.initialized = False
        self.generate_descriptions = generate_descriptions

    async def build_catalog(self, minio_storage, vector_store) -> None:
        """
        Строит каталог туров при старте приложения.
        """
        print("\n" + "=" * 60)
        print("Построение каталога туров...")
        print("=" * 60)

        try:
            documents = await minio_storage.list_documents()

            print(f"Найдено {len(documents)} документов в MinIO")

            # НЕ СОВПАДАЕТ document_id!! 

            for idx, doc in enumerate(documents, 1):
                document_id = doc['document_id']
                filename = doc['filename']

                if document_id in self.catalog:
                    continue

                print(f"  [{idx}/{len(documents)}] Обработка: {filename}")

                try:
                    chunks = await self._get_document_chunks(vector_store, filename, limit=2)

                    description = None

                    if self.generate_descriptions and chunks:
                        try:
                            description = await self._generate_description(filename, chunks)
                            print(f"    ✓ Описание создано: {description[:60]}...")
                        except Exception as desc_error:
                            print(f"    ⚠ Не удалось создать описание: {desc_error}")
                            description = None

                    self.catalog[document_id] = {
                        "document_id": document_id,
                        "filename": filename,
                        "tour_name": self._extract_tour_name(filename),
                        "description": description,
                        "chunks_count": len(chunks) if chunks else 0
                    }

                    if not chunks:
                        print(f"Документ не проиндексирован")
                    elif not self.generate_descriptions:
                        print(f"Добавлен без описания")

                except Exception as e:
                    print(f"    ✗ Ошибка при обработке: {e}")
                    self.catalog[document_id] = {
                        "document_id": document_id,
                        "filename": filename,
                        "tour_name": self._extract_tour_name(filename),
                        "description": None,
                        "chunks_count": 0
                    }

            self.initialized = True
            print(f"\nКаталог туров построен: {len(self.catalog)} туров")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\nОшибка при построении каталога: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 60 + "\n")

    async def _get_document_chunks(self, vector_store, filename: str, limit: int = 2) -> List[str]:
        """
        Получает первые N чанков документа из Qdrant по имени файла (source).
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            scroll_result = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=filename)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            chunks = []
            for point in scroll_result[0]:
                if "text" in point.payload:
                    chunks.append(point.payload["text"])

            return chunks
        except Exception as e:
            print(f"Ошибка при получении чанков: {e}")
            return []

    async def _generate_description(self, filename: str, chunks: List[str]) -> str:
        """
        Генерирует краткое описание тура через LLM на основе первых чанков.
        """
        try:
            context = "\n".join(chunks[:2])

            prompt = f"""Ты эксперт по туристическим программам. Перед тобой фрагмент описания тура из документа "{filename}".

ФРАГМЕНТ ДОКУМЕНТА:
{context[:1000]}

ЗАДАЧА: Создай КРАТКОЕ описание тура (1-2 предложения, максимум 100 слов).

ПРАВИЛА:
- Укажи направление/место
- Укажи тип тура (пляжный, экскурсионный, горнолыжный и т.д.)
- Укажи ключевые особенности если есть
- Будь лаконичен и информативен
- НЕ упоминай название файла
- Пиши в настоящем времени

ПРИМЕРЫ:
"Экскурсионный тур в Дагестан с посещением древних крепостей, каспийского побережья и горных аулов. Включает проживание, питание и трансферы."

"Пляжный отдых в Турции на курорте Анталья. All inclusive, 7 ночей, прямой перелет."

Верни ТОЛЬКО текст описания без дополнительных пояснений:"""

            description = await app_state.llm_client.simple_query(prompt)

            description = description.strip().strip('"').strip("'")

            if len(description) > 200:
                description = description[:197] + "..."

            return description

        except Exception as e:
            print(f"      Ошибка при генерации описания: {e}")
            return "Информация о туре доступна по запросу."

    def _extract_tour_name(self, filename: str) -> str:
        """
        Извлекает и стандартизирует название тура из имени файла.
        """
        name = filename.replace(".pdf", "").replace(".docx", "").replace(".PDF", "").replace(".DOCX", "")

        name = name.replace("Копия ", "").replace("копия ", "")
        name = name.replace("tmp", "").replace("TMP", "")

        name = name.replace("_", " ").replace("-", " ")

        import re
        name = re.sub(r'\s+', ' ', name)

        name = name.strip()

        words = name.split()
        standardized_words = []

        for word in words:
            if word.isupper() and 2 <= len(word) <= 4:
                standardized_words.append(word)
            elif word.isdigit():
                standardized_words.append(word)
            elif any(c.isdigit() for c in word):
                standardized_words.append(word)
            else:
                standardized_words.append(word.capitalize())

        name = " ".join(standardized_words)

        return name if name else "Тур без названия"

    def get_tour(self, document_id: str) -> Optional[Dict]:
        """
        Возвращает информацию о туре по document_id.
        """
        return self.catalog.get(document_id)

    def get_all_tours(self) -> List[Dict]:
        """
        Возвращает список всех туров.
        """
        return sorted(
            self.catalog.values(),
            key=lambda x: x.get('tour_name', '')
        )

    def filter_tours(self, destinations: List[str] = None, tour_types: List[str] = None) -> List[Dict]:
        """
        Фильтрует туры по направлениям и типам.

        Args:
            destinations: список направлений/стран для фильтрации (например: ["Япония", "Азия"])
            tour_types: список типов туров (например: ["пляжный", "экскурсионный"])

        Returns:
            Список отфильтрованных туров
        """
        if not destinations and not tour_types:
            return self.get_all_tours()

        filtered = []

        for tour in self.catalog.values():
            tour_name = tour.get('tour_name', '').lower()
            description = (tour.get('description') or '').lower()
            filename = tour.get('filename', '').lower()

            search_text = f"{tour_name} {description} {filename}"

            matches_destination = False
            if destinations:
                for dest in destinations:
                    dest_lower = dest.lower()
                    if dest_lower in search_text:
                        matches_destination = True
                        break
            else:
                matches_destination = True

            matches_tour_type = False
            if tour_types:
                for tour_type in tour_types:
                    tour_type_lower = tour_type.lower()
                    if tour_type_lower in search_text:
                        matches_tour_type = True
                        break
            else:
                matches_tour_type = True

            if matches_destination and matches_tour_type:
                filtered.append(tour)

        return sorted(filtered, key=lambda x: x.get('tour_name', ''))

    def format_catalog(self, include_descriptions: bool = True) -> str:
        """
        Форматирует каталог туров в читаемый вид.
        """
        tours = self.get_all_tours()

        if not tours:
            return "Каталог туров пуст."

        lines = []
        for idx, tour in enumerate(tours, 1):
            tour_name = tour.get('tour_name', 'Неизвестный тур')
            description = tour.get('description')

            if include_descriptions and description:
                lines.append(f"{idx}. **{tour_name}**")
                lines.append(f"   {description}")
                lines.append("")
            else:
                lines.append(f"{idx}. {tour_name}")

        return "\n".join(lines)
