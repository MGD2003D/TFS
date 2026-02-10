from typing import List, Dict, Any
import re
from collections import Counter
import math


class BM25SparseEncoder:
    """
    BM25-based sparse vector encoder для hybrid search.

    Создает sparse vectors на основе BM25 алгоритма для точного поиска по ключевым словам.
    Комбинируется с dense vectors для hybrid search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Параметр насыщения термина (обычно 1.2-2.0)
            b: Параметр нормализации длины документа (обычно 0.75)
        """
        self.k1 = k1
        self.b = b
        self.vocab = {}  # word -> index
        self.idf = {}    # word -> idf score
        self.avg_doc_len = 0
        self.doc_count = 0

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст.

        Для русского языка используем простую токенизацию по словам,
        приводим к нижнему регистру, убираем пунктуацию.
        """
        text = text.lower()

        words = re.findall(r'\b\w+\b', text, re.UNICODE)

        words = [w for w in words if len(w) >= 2]

        return words

    def build_vocab(self, documents: List[str]) -> None:
        """
        Строит словарь и IDF из документов.

        Args:
            documents: Список текстов документов
        """
        tokenized_docs = [self.tokenize(doc) for doc in documents]

        all_words = set()
        for tokens in tokenized_docs:
            all_words.update(tokens)

        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}

        self.doc_count = len(documents)
        doc_freq = Counter()

        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        for word, df in doc_freq.items():
            self.idf[word] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

        total_len = sum(len(tokens) for tokens in tokenized_docs)
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0

        print(f"[BM25] Built vocabulary: {len(self.vocab)} unique words")
        print(f"[BM25] Average document length: {self.avg_doc_len:.2f} tokens")

    def encode(self, text: str) -> Dict[int, float]:
        """
        Кодирует текст в sparse vector (словарь индекс -> вес).

        Args:
            text: Текст для кодирования

        Returns:
            Sparse vector как словарь {index: weight}
        """
        tokens = self.tokenize(text)
        doc_len = len(tokens)

        term_freq = Counter(tokens)

        sparse_vector = {}

        for term, freq in term_freq.items():
            if term in self.vocab:
                # BM25 формула
                idf = self.idf.get(term, 0)

                # Нормализованная частота термина
                tf = (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                )

                weight = idf * tf

                if weight > 0:
                    sparse_vector[self.vocab[term]] = weight

        return sparse_vector

    def encode_query(self, query: str, top_k: int = 100) -> Dict[int, float]:
        """
        Кодирует поисковый запрос в sparse vector.

        Для запросов используем упрощенную схему (IDF взвешивание без BM25).

        Args:
            query: Поисковый запрос
            top_k: Максимальное количество термов в sparse vector

        Returns:
            Sparse vector как словарь {index: weight}
        """
        tokens = self.tokenize(query)
        term_freq = Counter(tokens)

        sparse_vector = {}

        for term, freq in term_freq.items():
            if term in self.vocab:
                idf = self.idf.get(term, 0)
                weight = idf * freq

                if weight > 0:
                    sparse_vector[self.vocab[term]] = weight

        if len(sparse_vector) > top_k:
            sorted_items = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)
            sparse_vector = dict(sorted_items[:top_k])

        return sparse_vector

    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return len(self.vocab)
