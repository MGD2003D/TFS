from pypdf import PdfReader
from typing import List, Dict, Any
import os


class DocumentIndexer:

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

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

    async def process_pdf(self, pdf_path: str) -> tuple[List[str], List[Dict[str, Any]]]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)

        metadata = []
        for idx, chunk in enumerate(chunks):
            metadata.append({
                "source": os.path.basename(pdf_path),
                "chunk_id": idx,
                "total_chunks": len(chunks)
            })

        print(f"Извлечено {len(chunks)} чанков из {pdf_path}")
        return chunks, metadata

    async def process_multiple_pdfs(self, pdf_paths: List[str]) -> tuple[List[str], List[Dict[str, Any]]]:
        all_chunks = []
        all_metadata = []

        for pdf_path in pdf_paths:
            chunks, metadata = await self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

        return all_chunks, all_metadata
