from pypdf import PdfReader
from docx import Document
from typing import List, Dict, Any, Optional
import os
import re
import time
from services.document_types import normalize_extension
from services.ner_service import extract_entities


# ─── Email parsing ─────────────────────────────────────────────────────────────

# Паттерны заголовков письма (RU + EN)
_EMAIL_HEADERS = {
    "from":    re.compile(r'(?:От|From|Отправитель|Sender)\s*:\s*(.+)', re.IGNORECASE),
    "to":      re.compile(r'(?:Кому|To|Получатель)\s*:\s*(.+)', re.IGNORECASE),
    "cc":      re.compile(r'(?:Копия|CC|Cc)\s*:\s*(.+)', re.IGNORECASE),
    "date":    re.compile(r'(?:Дата|Date|Отправлено|Sent)\s*:\s*(.+)', re.IGNORECASE),
    "subject": re.compile(r'(?:Тема|Subject|Re:|Fwd:)\s*:\s*(.+)', re.IGNORECASE),
}

# Разделители между письмами в цепочке
_CHAIN_SEPARATOR = re.compile(
    r'(?:'
    r'-{5,}|={5,}|\*{5,}|_{5,}'
    r'|_{3,}\s*Forwarded\s+message'
    r'|-{3,}\s*(?:Original|Forwarded)'
    r'|(?:^|\n)(?:От|From)\s*:'
    r')',
    re.IGNORECASE | re.MULTILINE
)


def extract_email_metadata(text: str) -> Dict[str, str]:
    """
    Извлекает поля заголовка из текста письма.
    Возвращает dict с найденными полями (пустой если это не письмо).
    """
    meta = {}
    # Ищем только в первых 50 строках — заголовки обычно вверху
    header_zone = "\n".join(text.splitlines()[:50])
    for field, pattern in _EMAIL_HEADERS.items():
        m = pattern.search(header_zone)
        if m:
            meta[field] = m.group(1).strip()[:200]
    return meta


def split_email_chain(text: str) -> List[str]:
    """
    Разбивает текст PDF с цепочкой писем на отдельные письма.
    Если разделителей нет — возвращает [text] (одно письмо).
    """
    parts = _CHAIN_SEPARATOR.split(text)
    # Отфильтровываем слишком короткие фрагменты (< 100 символов)
    result = [p.strip() for p in parts if len(p.strip()) >= 100]
    return result if result else [text]


def format_email_header(meta: Dict[str, str]) -> str:
    """Строит читаемую шапку письма для включения в чанк."""
    if not meta:
        return ""
    parts = []
    if meta.get("date"):
        parts.append(f"Дата: {meta['date']}")
    if meta.get("from"):
        parts.append(f"От: {meta['from']}")
    if meta.get("to"):
        parts.append(f"Кому: {meta['to']}")
    if meta.get("cc"):
        parts.append(f"Копия: {meta['cc']}")
    if meta.get("subject"):
        parts.append(f"Тема: {meta['subject']}")
    return "\n".join(parts)


def _ner_progress(idx: int, total: int, t0: float) -> None:
    step = max(1, min(50, total // 10))
    if (idx + 1) % step == 0 or idx + 1 == total:
        elapsed = time.time() - t0
        pct = (idx + 1) * 100 // total
        filled = pct // 5
        bar = "█" * filled + "░" * (20 - filled)
        print(f"  [NER] [{bar}] {pct:3d}% ({idx+1}/{total})  {elapsed:.1f}s elapsed", flush=True)


# ─── Indexer ───────────────────────────────────────────────────────────────────

class DocumentIndexer:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def extract_text_from_docx(self, docx_path: str) -> str:
        doc = Document(docx_path)
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text_parts.append(paragraph.text)
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    text_parts.append("\t".join(row_cells))
        return "\n".join(text_parts)

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return [c for c in chunks if c]

    def _build_email_chunks(
        self,
        full_text: str,
        document_id: str,
        source_name: str,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Разбивает текст на письма → чанкует каждое письмо.
        Каждый чанк получает метаданные письма (from, to, date, subject).
        Шапка письма prepend-ится к первому чанку каждого письма.
        """
        emails = split_email_chain(full_text)
        is_chain = len(emails) > 1
        print(f"  [EMAIL] Найдено писем в цепочке: {len(emails)}")

        all_chunks: List[str] = []
        all_meta: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for email_idx, email_text in enumerate(emails):
            meta = extract_email_metadata(email_text)
            header_str = format_email_header(meta)
            chunks = self.chunk_text(email_text)

            for local_idx, chunk in enumerate(chunks):
                # Prepend заголовок к первому чанку письма
                if local_idx == 0 and header_str:
                    chunk = f"[Письмо {email_idx + 1}]\n{header_str}\n\n{chunk}"

                all_chunks.append(chunk)
                all_meta.append({
                    "document_id":  document_id,
                    "source":       source_name,
                    "chunk_id":     global_chunk_idx,
                    "total_chunks": None,  # заполним после
                    "email_index":  email_idx,
                    "email_count":  len(emails),
                    "is_chain":     is_chain,
                    "email_from":   meta.get("from", ""),
                    "email_to":     meta.get("to", ""),
                    "email_cc":     meta.get("cc", ""),
                    "email_date":   meta.get("date", ""),
                    "email_subject": meta.get("subject", ""),
                })
                global_chunk_idx += 1

        # Проставляем total_chunks
        total = len(all_chunks)
        for m in all_meta:
            m["total_chunks"] = total

        return all_chunks, all_meta

    async def process_pdf(
        self,
        pdf_path: str,
        document_id: str = None,
        original_filename: str = None,
        is_email: bool = True,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

        text = self.extract_text_from_pdf(pdf_path)

        if document_id is None:
            import hashlib
            document_id = hashlib.sha256(os.path.basename(pdf_path).encode()).hexdigest()[:16]

        source_name = original_filename if original_filename else os.path.basename(pdf_path)

        if is_email:
            chunks, metadata = self._build_email_chunks(text, document_id, source_name)
        else:
            chunks = self.chunk_text(text)
            metadata = [
                {
                    "document_id":  document_id,
                    "source":       source_name,
                    "chunk_id":     idx,
                    "total_chunks": len(chunks),
                }
                for idx in range(len(chunks))
            ]

        print(f"  Извлечено {len(chunks)} чанков из {source_name}")

        # NER
        print(f"  [NER] Начинаю обработку {len(chunks)} чанков...", flush=True)
        t0 = time.time()
        for idx, (chunk, meta) in enumerate(zip(chunks, metadata)):
            ner = extract_entities(chunk)
            meta["entity_texts"] = ner["entity_texts"]
            meta["entity_labels"] = ner["entity_labels"]
            _ner_progress(idx, len(chunks), t0)
        print(f"  [NER] Готово за {time.time() - t0:.1f}s", flush=True)

        return chunks, metadata

    async def process_docx(
        self,
        docx_path: str,
        document_id: str = None,
        original_filename: str = None,
        is_email: bool = True,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"File not found: {docx_path}")

        text = self.extract_text_from_docx(docx_path)

        if document_id is None:
            import hashlib
            document_id = hashlib.sha256(os.path.basename(docx_path).encode()).hexdigest()[:16]

        source_name = original_filename if original_filename else os.path.basename(docx_path)

        if is_email:
            chunks, metadata = self._build_email_chunks(text, document_id, source_name)
        else:
            chunks = self.chunk_text(text)
            metadata = [
                {
                    "document_id":  document_id,
                    "source":       source_name,
                    "chunk_id":     idx,
                    "total_chunks": len(chunks),
                }
                for idx in range(len(chunks))
            ]

        print(f"  Extracted {len(chunks)} chunks from {source_name}")

        print(f"  [NER] Начинаю обработку {len(chunks)} чанков...", flush=True)
        t0 = time.time()
        for idx, (chunk, meta) in enumerate(zip(chunks, metadata)):
            ner = extract_entities(chunk)
            meta["entity_texts"] = ner["entity_texts"]
            meta["entity_labels"] = ner["entity_labels"]
            _ner_progress(idx, len(chunks), t0)
        print(f"  [NER] Готово за {time.time() - t0:.1f}s", flush=True)

        return chunks, metadata

    async def process_document(
        self,
        file_path: str,
        document_id: str = None,
        original_filename: str = None,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        ext = normalize_extension(file_path)
        if ext == ".pdf":
            return await self.process_pdf(file_path, document_id=document_id, original_filename=original_filename)
        if ext == ".docx":
            return await self.process_docx(file_path, document_id=document_id, original_filename=original_filename)
        raise ValueError(f"Unsupported document type: {ext}")

    async def process_multiple_documents(
        self,
        file_paths: List[str],
        document_ids: List[str] = None,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        all_chunks, all_metadata = [], []
        for idx, file_path in enumerate(file_paths):
            document_id = document_ids[idx] if document_ids and idx < len(document_ids) else None
            chunks, metadata = await self.process_document(file_path, document_id=document_id)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
        return all_chunks, all_metadata

    async def process_multiple_pdfs(
        self,
        pdf_paths: List[str],
        document_ids: List[str] = None,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        return await self.process_multiple_documents(pdf_paths, document_ids=document_ids)
