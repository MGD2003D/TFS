"""
BGE-M3 Encoder for hybrid search (dense + sparse).

Uses BAAI/bge-m3 model which provides both dense and sparse embeddings
in a single forward pass, eliminating the need for separate BM25 vocab.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import os


class BGEm3Encoder:
    """
    BGE-M3 encoder for hybrid dense + sparse retrieval.

    Provides:
    - Dense embeddings (1024-dim) for semantic search
    - Sparse embeddings (lexical matching, learned weights)
    - Both from single model, no vocab needed

    Based on BAAI/bge-m3 (https://arxiv.org/abs/2402.03216)
    Used in PankRAG paper as baseline.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None, cache_dir: str = None):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu' (auto-detected if None)
            cache_dir: Cache directory for models (uses HF_HOME env var if None)
        """
        self.model_name = model_name

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Setup cache directory (prefer HF_HOME to avoid deprecation warning)
        if cache_dir is None:
            cache_dir = os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE') or os.path.expanduser('~/.cache/huggingface')

        print(f"[BGE-M3] Loading {model_name} on {device.upper()}...")
        print(f"[BGE-M3] Cache directory: {cache_dir}")
        print("[BGE-M3] This may take a few minutes on first run (~2GB download)...")

        try:
            from FlagEmbedding import BGEM3FlagModel
            # Set cache directory for HuggingFace (use HF_HOME to avoid deprecation warning)
            if 'HF_HOME' not in os.environ:
                os.environ['HF_HOME'] = cache_dir
            self.model = BGEM3FlagModel(model_name, use_fp16=torch.cuda.is_available())
            print(f"[BGE-M3] Model loaded successfully!")
        except ImportError as e:
            import sys
            error_msg = (
                f"FlagEmbedding not installed.\n"
                f"Python executable: {sys.executable}\n"
                f"Python version: {sys.version}\n"
                f"sys.path: {sys.path}\n"
                f"Original error: {e}\n\n"
                f"Install with: python -m pip install -U FlagEmbedding"
            )
            raise ImportError(error_msg)

    def _convert_sparse_keys(self, sparse_dict: dict) -> Dict[int, float]:
        """
        Convert BGE-M3 sparse dict keys to integer token IDs.

        BGE-M3 returns keys as strings like '581', '125921' (string token IDs).
        We need to convert them to integers for Qdrant sparse vectors.
        """
        return {int(token): float(weight) for token, weight in sparse_dict.items()}

    def encode_dense(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode texts to dense embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2-normalize (compatibility param, BGE-M3 already normalizes)

        Returns:
            Dense embeddings (num_texts, 1024)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']

        embeddings = np.array(embeddings)

        # BGE-M3 already returns normalized embeddings, but apply L2 norm if explicitly requested
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)  # Avoid division by zero

        return embeddings

    def encode_sparse(self, texts: List[str], batch_size: int = 32) -> List[Dict[int, float]]:
        """
        Encode texts to sparse embeddings (lexical matching).

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            List of sparse vectors as {token_id: weight} dicts
        """
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )

        sparse_vecs = []
        for idx, sparse_dict in enumerate(result['lexical_weights']):
            # DEBUG: Check format on first item
            if idx == 0 and sparse_dict:
                sample_key = list(sparse_dict.keys())[0]
                print(f"[BGE-M3 DEBUG] Sparse key type: {type(sample_key)}, sample: {sample_key}")
                print(f"[BGE-M3 DEBUG] Sample keys: {list(sparse_dict.keys())[:5]}")

            # Convert string token IDs to integers
            sparse_vec = self._convert_sparse_keys(sparse_dict)
            sparse_vecs.append(sparse_vec)

        return sparse_vecs

    def encode_hybrid(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        """
        Encode texts to both dense and sparse embeddings in single pass.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            (dense_embeddings, sparse_vectors) tuple
        """
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        dense_vecs = np.array(result['dense_vecs'])

        # Convert sparse to Qdrant format (string token IDs -> int)
        sparse_vecs = [self._convert_sparse_keys(sparse_dict) for sparse_dict in result['lexical_weights']]

        return dense_vecs, sparse_vecs

    def get_dense_dim(self) -> int:
        """Get dense embedding dimension (1024 for bge-m3)."""
        return 1024

    # Compatibility methods for drop-in replacement of BM25SparseEncoder
    def build_vocab(self, documents: List[str]) -> None:
        """
        No-op for BGE-M3 (vocab not needed, learned sparse).

        Kept for backward compatibility with BM25SparseEncoder interface.
        """
        print("[BGE-M3] Vocab building not needed (learned sparse), skipping...")

    def encode(self, sentences, batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = True, **kwargs):
        """
        Universal encode method for compatibility with both:
        - SentenceTransformer API (returns dense embeddings as np.ndarray)
        - BM25SparseEncoder API (returns sparse vector as Dict[int, float])

        Args:
            sentences: str (returns sparse) or List[str] (returns dense)
            batch_size: Batch size for encoding
            show_progress_bar: Ignored (compatibility param)
            normalize_embeddings: Whether to normalize dense embeddings
            **kwargs: Other compatibility params (ignored)

        Returns:
            - If sentences is str: Dict[int, float] (sparse vector)
            - If sentences is List[str]: np.ndarray (dense embeddings)
        """
        # Single string -> sparse vector (for sparse_encoder.encode(text))
        if isinstance(sentences, str):
            sparse_vecs = self.encode_sparse([sentences], batch_size=1)
            return sparse_vecs[0]

        # List of strings -> dense embeddings (for embedding_model.encode(texts))
        return self.encode_dense(sentences, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    def encode_query(self, query: str, top_k: int = 100) -> Dict[int, float]:
        """
        Encode query to sparse vector.

        Compatibility method for BM25SparseEncoder interface.
        Note: top_k parameter ignored (BGE-M3 returns variable-length sparse).
        """
        return self.encode(query)

    def get_vocab_size(self) -> int:
        """
        Return vocab size (not applicable for BGE-M3).

        Returns large number for compatibility.
        """
        return 2**31  # Theoretical max from hash space
