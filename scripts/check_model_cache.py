"""
Утилита для проверки и управления кэшем моделей HuggingFace.

Usage:
    python scripts/check_model_cache.py          # Показать информацию о кэше
    python scripts/check_model_cache.py --clear   # Очистить кэш (осторожно!)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


def get_cache_dirs() -> List[Tuple[str, Path]]:
    """Получить все директории кэша."""
    cache_dirs = []

    # HuggingFace transformers cache
    transformers_cache = os.getenv('TRANSFORMERS_CACHE')
    if not transformers_cache:
        transformers_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
    else:
        transformers_cache = Path(transformers_cache)

    if transformers_cache.exists():
        cache_dirs.append(("Transformers", transformers_cache))

    # Sentence-transformers cache
    st_cache = os.getenv('SENTENCE_TRANSFORMERS_HOME')
    if not st_cache:
        st_cache = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
    else:
        st_cache = Path(st_cache)

    if st_cache.exists():
        cache_dirs.append(("Sentence-Transformers", st_cache))

    # HF_HOME (общий кэш)
    hf_home = os.getenv('HF_HOME')
    if hf_home and Path(hf_home).exists():
        cache_dirs.append(("HF_HOME", Path(hf_home)))

    return cache_dirs


def get_dir_size(path: Path) -> int:
    """Получить размер директории в байтах."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"  Ошибка при подсчете размера {path}: {e}")
    return total


def format_size(bytes: int) -> str:
    """Форматировать размер в человекочитаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def list_models(cache_dir: Path) -> List[str]:
    """Список моделей в кэше."""
    models = []
    try:
        if cache_dir.name == 'hub':
            # HuggingFace hub format: models--org--name
            for item in cache_dir.iterdir():
                if item.is_dir() and item.name.startswith('models--'):
                    model_name = item.name.replace('models--', '').replace('--', '/')
                    models.append(model_name)
        elif cache_dir.name == 'sentence_transformers':
            # Sentence-transformers format: org_name
            for item in cache_dir.iterdir():
                if item.is_dir():
                    models.append(item.name.replace('_', '/'))
    except Exception as e:
        print(f"  Ошибка при чтении моделей из {cache_dir}: {e}")

    return sorted(models)


def show_cache_info():
    """Показать информацию о кэше."""
    print("=" * 80)
    print("ИНФОРМАЦИЯ О КЭШЕ МОДЕЛЕЙ")
    print("=" * 80)
    print()

    cache_dirs = get_cache_dirs()

    if not cache_dirs:
        print("⚠️  Кэш не найден!")
        print("\nПроверьте переменные окружения в .env:")
        print("  - HF_HOME")
        print("  - TRANSFORMERS_CACHE")
        print("  - SENTENCE_TRANSFORMERS_HOME")
        return

    total_size = 0

    for cache_type, cache_path in cache_dirs:
        print(f"📁 {cache_type}")
        print(f"   Путь: {cache_path}")

        size = get_dir_size(cache_path)
        total_size += size
        print(f"   Размер: {format_size(size)}")

        models = list_models(cache_path)
        if models:
            print(f"   Модели ({len(models)}):")
            for model in models[:10]:  # Показываем первые 10
                print(f"     - {model}")
            if len(models) > 10:
                print(f"     ... и ещё {len(models) - 10} моделей")
        else:
            print("   Модели: нет")

        print()

    print("=" * 80)
    print(f"ИТОГО: {format_size(total_size)}")
    print("=" * 80)


def clear_cache():
    """Очистить кэш (с подтверждением)."""
    print("⚠️  ВНИМАНИЕ! Это удалит ВСЕ скачанные модели.")
    print("Модели придется скачивать заново (~2-4 GB).")
    print()

    response = input("Продолжить? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Отменено.")
        return

    cache_dirs = get_cache_dirs()

    for cache_type, cache_path in cache_dirs:
        print(f"\nОчищаю {cache_type}: {cache_path}")
        try:
            import shutil
            shutil.rmtree(cache_path)
            print(f"✅ Удалено: {cache_path}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    print("\n✅ Кэш очищен!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--clear':
        clear_cache()
    else:
        show_cache_info()
