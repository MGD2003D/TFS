from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported_document(filename: str) -> bool:
    return normalize_extension(filename) in SUPPORTED_EXTENSIONS


def get_content_type(filename: str) -> str:
    ext = normalize_extension(filename)
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def temp_suffix_for(filename: str, default: str = ".bin") -> str:
    ext = normalize_extension(filename)
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    return default
