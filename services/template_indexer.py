import csv
import os
import tempfile
from typing import List, Dict

REQUEST_COL = "Вопросы"
RESPONSE_COL = "Целевой ответ (GOLD)"


class TemplateIndexer:

    def parse_csv(self, file_path: str) -> List[Dict]:
        templates = []
        with open(file_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_id, row in enumerate(reader):
                request = (row.get(REQUEST_COL) or "").strip()
                response = (row.get(RESPONSE_COL) or "").strip()
                if request and response:
                    templates.append({
                        "request": request,
                        "response": response,
                        "row_id": row_id,
                    })
        return templates

    async def index_file(self, file_path: str, source: str, vector_store) -> int:
        templates = self.parse_csv(file_path)
        if not templates:
            print(f"[TEMPLATES] Нет данных в {source}")
            return 0
        return await self.index_templates(templates, source, vector_store)

    async def index_templates(self, templates: List[Dict], source: str, vector_store) -> int:
        # Удаляем старые записи с тем же source
        try:
            await vector_store.delete_by_source(source)
        except Exception as e:
            print(f"[TEMPLATES] Не удалось удалить старые записи {source}: {e}")

        texts = [t["request"] for t in templates]
        metadata = [
            {
                "response": t["response"],
                "source": source,
                "row_id": t["row_id"],
                "type": "mail_template",
            }
            for t in templates
        ]

        await vector_store.add_documents(texts, metadata)
        print(f"[TEMPLATES] Проиндексировано {len(templates)} шаблонов из {source}")
        return len(templates)

    async def index_from_bytes(self, content: bytes, source: str, vector_store) -> int:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            tmp.write(content)
            tmp.close()
            return await self.index_file(tmp.name, source, vector_store)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
