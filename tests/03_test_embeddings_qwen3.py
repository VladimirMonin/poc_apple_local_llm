#!/usr/bin/env python3
"""
Тест Qwen3-Embedding-0.6B-4bit-DWQ (БЕЗ PyTorch!)

Эта модель:
- Использует ТОЛЬКО MLX (без PyTorch зависимостей)
- Мультиязычная (русский, английский, китайский, 100+ языков)
- 1024 измерения (больше чем 384 у all-MiniLM)
- ~335 MB размер (vs 150 MB у all-MiniLM)
- Загружается через mlx-lm (чистый MLX стек)
"""

import sys
import time
import numpy as np
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.profiler import SystemProfiler
import mlx.core as mx
from mlx_lm import load


def get_embeddings_qwen3(texts: list, model, tokenizer) -> mx.array:
    """
    Получить embeddings через Qwen3 модель (БЕЗ PyTorch!)

    Процесс:
    1. Токенизация через mlx-lm tokenizer
    2. Прямой проход через transformer слои (MLX)
    3. Mean pooling по sequence dimension
    4. Возврат MLX array (не numpy, не torch!)
    """
    embeddings = []

    for text in texts:
        # Токенизируем
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])

        # Получаем hidden states (прямой доступ к слоям MLX)
        h = model.model.embed_tokens(input_ids)
        for layer in model.model.layers:
            h = layer(h, mask=None, cache=None)
        h = model.model.norm(h)

        # Mean pooling
        pooled = mx.mean(h, axis=1)  # [1, 1024]
        mx.eval(pooled)  # Форсируем вычисление

        embeddings.append(pooled[0])  # Берем первый элемент батча

    # Стакаем в один массив
    return mx.stack(embeddings)


def cosine_similarity_mlx(a: mx.array, b: mx.array) -> float:
    """Косинусное сходство для MLX arrays"""
    return float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b))))


def test_model_loading():
    """Тест 1: Загрузка модели через mlx-lm (БЕЗ PyTorch!)"""
    print("\n" + "=" * 80)
    print("ТЕСТ 1: ЗАГРУЗКА QWEN3-EMBEDDING-0.6B-4BIT-DWQ")
    print("=" * 80)
    print()
    print("🎯 Ключевое отличие: загружается через mlx-lm (чистый MLX стек)")
    print("   ✅ БЕЗ PyTorch")
    print("   ✅ БЕЗ sentence-transformers")
    print("   ✅ Только MLX зависимости")
    print()

    profiler = SystemProfiler()

    start_time = time.time()
    model, tokenizer = load("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    load_time = time.time() - start_time

    mem_snapshot = profiler._get_memory_snapshot()

    print(f"✅ Модель загружена за {load_time:.2f} сек")
    print(f"   Тип модели: {type(model).__name__}")
    print(f"   Тип токенизатора: {type(tokenizer).__name__}")
    print()
    print("📊 МЕТРИКИ ЗАГРУЗКИ:")
    print("-" * 80)
    print(f"   RAM процесса: {mem_snapshot.rss_mb:.1f} MB")
    print(f"   Swap: {mem_snapshot.swap_used_mb:.1f} MB")
    print("-" * 80)

    return model, tokenizer


def test_multilingual(model, tokenizer):
    """Тест 2: Мультиязычность (русский, английский, китайский)"""
    print("\n" + "=" * 80)
    print("ТЕСТ 2: МУЛЬТИЯЗЫЧНОСТЬ (РУССКИЙ + АНГЛИЙСКИЙ + КИТАЙСКИЙ)")
    print("=" * 80)
    print()

    test_pairs = [
        {
            "name": "Русский vs Английский (одна тема - ML)",
            "text1": "Машинное обучение и искусственный интеллект",
            "text2": "Machine learning and artificial intelligence",
            "expected_similarity": "высокое (>0.25)",
        },
        {
            "name": "Русский vs Китайский (одна тема - ML)",
            "text1": "Машинное обучение и искусственный интеллект",
            "text2": "机器学习和人工智能",
            "expected_similarity": "высокое (>0.35)",
        },
        {
            "name": "Английский vs Китайский (одна тема - ML)",
            "text1": "Machine learning and artificial intelligence",
            "text2": "机器学习和人工智能",
            "expected_similarity": "высокое (>0.35)",
        },
        {
            "name": "Русский (ML) vs Русский (погода)",
            "text1": "Машинное обучение использует нейронные сети",
            "text2": "Сегодня хорошая солнечная погода",
            "expected_similarity": "низкое (<0.3)",
        },
        {
            "name": "Английский (ML) vs Английский (погода)",
            "text1": "Machine learning uses neural networks",
            "text2": "Today is a beautiful sunny day",
            "expected_similarity": "низкое (<0.3)",
        },
        {
            "name": "Китайский (ML) vs Китайский (погода)",
            "text1": "机器学习使用神经网络",
            "text2": "今天天气很好",
            "expected_similarity": "низкое (<0.3)",
        },
    ]

    results = []
    profiler = SystemProfiler()

    for pair in test_pairs:
        print(f"📝 {pair['name']}")
        print(f"   Текст 1: {pair['text1']}")
        print(f"   Текст 2: {pair['text2']}")

        start_time = time.time()
        embeddings = get_embeddings_qwen3(
            [pair["text1"], pair["text2"]], model, tokenizer
        )
        embed_time = time.time() - start_time

        similarity = cosine_similarity_mlx(embeddings[0], embeddings[1])

        print(
            f"   Сходство: {similarity:.4f} (ожидалось {pair['expected_similarity']})"
        )
        print(f"   Время: {embed_time:.3f} сек")
        print()

        results.append(
            {
                "name": pair["name"],
                "similarity": similarity,
                "time": embed_time,
                "embedding_dim": embeddings.shape[1],
            }
        )

    mem_snapshot = profiler._get_memory_snapshot()

    print("📊 СВОДКА ПО МУЛЬТИЯЗЫЧНОСТИ:")
    print("-" * 80)
    print(f"   Средняя скорость: {np.mean([r['time'] for r in results]):.3f} сек")
    print(f"   Размерность векторов: {results[0]['embedding_dim']}")
    print(f"   RAM: {mem_snapshot.rss_mb:.1f} MB")
    print(f"   Swap: {mem_snapshot.swap_used_mb:.1f} MB")
    print("-" * 80)

    return results


def test_code_understanding(model, tokenizer):
    """Тест 3: Понимание кода (важно для RAG)"""
    print("\n" + "=" * 80)
    print("ТЕСТ 3: ПОНИМАНИЕ КОДА")
    print("=" * 80)
    print()

    test_pairs = [
        {
            "name": "Похожие функции (Python)",
            "code1": """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
""",
            "code2": """
def add_all(values):
    result = 0
    for v in values:
        result = result + v
    return result
""",
            "expected_similarity": "высокое (>0.7)",
        },
        {
            "name": "Похожие концепции (разные языки)",
            "code1": """
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
""",
            "code2": """
function bubbleSort(array) {
    for (let i = 0; i < array.length; i++) {
        for (let j = 0; j < array.length - 1; j++) {
            if (array[j] > array[j+1]) {
                [array[j], array[j+1]] = [array[j+1], array[j]];
            }
        }
    }
}
""",
            "expected_similarity": "среднее (>0.5)",
        },
        {
            "name": "Разные концепции",
            "code1": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
            "code2": """
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}
""",
            "expected_similarity": "низкое (<0.4)",
        },
    ]

    results = []

    for pair in test_pairs:
        print(f"💻 {pair['name']}")

        start_time = time.time()
        embeddings = get_embeddings_qwen3(
            [pair["code1"], pair["code2"]], model, tokenizer
        )
        embed_time = time.time() - start_time

        similarity = cosine_similarity_mlx(embeddings[0], embeddings[1])

        print(
            f"   Сходство кода: {similarity:.4f} (ожидалось {pair['expected_similarity']})"
        )
        print(f"   Время: {embed_time:.3f} сек")
        print()

        results.append(
            {
                "name": pair["name"],
                "similarity": similarity,
                "time": embed_time,
            }
        )

    profiler = SystemProfiler()
    mem_snapshot = profiler._get_memory_snapshot()

    print("-" * 80)
    print(f"   Средняя скорость: {np.mean([r['time'] for r in results]):.3f} сек")
    print(f"   RAM: {mem_snapshot.rss_mb:.1f} MB")
    print(f"   Swap: {mem_snapshot.swap_used_mb:.1f} MB")
    print("-" * 80)

    return results


def print_final_comparison():
    """Итоговое сравнение"""
    print("\n" + "=" * 80)
    print("ИТОГОВОЕ СРАВНЕНИЕ: Qwen3-Embedding vs all-MiniLM vs BGE-small")
    print("=" * 80)
    print()

    comparison = """
┌─────────────────────────┬──────────────────┬─────────────────────────┬─────────────────────────┐
│ Параметр                │ Qwen3-0.6B       │ all-MiniLM-L6-v2-4bit   │ BGE-small-en-v1.5-4bit  │
├─────────────────────────┼──────────────────┼─────────────────────────┼─────────────────────────┤
│ Размерность векторов    │ 1024             │ 384                     │ 384                     │
│ Размер модели           │ ~335 MB          │ ~150 MB                 │ ~19 MB                  │
│ Формат загрузки         │ mlx-lm           │ mlx-embeddings          │ mlx-embeddings          │
│ PyTorch зависимость     │ ❌ НЕТ!          │ ❌ НЕТ                  │ ❌ НЕТ                  │
│ Русский язык            │ ✅ Отлично       │ ✅ Хорошо               │ ❌ ПРОВАЛ (0.70)        │
│ Английский язык         │ ✅ Отлично       │ ✅ Хорошо               │ ✅ Хорошо               │
│ Китайский язык          │ ✅ ДА!           │ ❌ Нет                  │ ❌ Нет                  │
│ Всего языков            │ 100+             │ ~50                     │ 1 (English)             │
│ Качество (код)          │ ✅ Отлично       │ ✅ Хорошо               │ ⚠️ Приемлемо            │
│ Скорость (4 текста)     │ ~0.4 сек         │ ~0.1 сек                │ ~0.02 сек               │
│ Макс. контекст          │ 8192 токена      │ 512 токенов             │ 512 токенов             │
│ Benchmark Score (MTEB)  │ 64.33            │ ~58                     │ ~55                     │
└─────────────────────────┴──────────────────┴─────────────────────────┴─────────────────────────┘

КЛЮЧЕВЫЕ ПРЕИМУЩЕСТВА QWEN3-EMBEDDING:

    ✅ ЧИСТЫЙ MLX СТЕК
       - Загружается через mlx-lm (не через mlx-embeddings)
       - БЕЗ PyTorch зависимостей вообще!
       - Прямой доступ к transformer слоям через MLX API
       - Идеальная интеграция с MLX экосистемой

    ✅ МУЛЬТИЯЗЫЧНОСТЬ
       - 100+ языков из коробки
       - Русский + Английский + Китайский (наша цель!)
       - Кросс-языковое понимание (RUS-CHN: 0.41)

    ✅ БОЛЬШАЯ РАЗМЕРНОСТЬ
       - 1024 vs 384 у конкурентов
       - Больше информации в векторе
       - Лучшее качество на сложных задачах

    ✅ ДЛИННЫЙ КОНТЕКСТ
       - 8192 токена vs 512 у других
       - Можно обрабатывать большие документы

    ⚠️ НЕДОСТАТКИ:
       - Медленнее (0.4 сек vs 0.1 сек)
       - Больше памяти (335 MB vs 150 MB)
       - Для 8 GB RAM может быть критично

ПОЧЕМУ ЭТО ВАЖНО ДЛЯ НАС:

    1. БЕЗ PyTorch = меньше зависимостей, проще развертывание
    2. Мультиязычность = наша ключевая потребность (RUS+ENG+CHN)
    3. Прямой MLX = лучшая интеграция с остальным стеком (Qwen3-VL, Whisper)
    4. Длинный контекст = больше информации в RAG
    5. Высокое качество = лучшие результаты поиска

ВЫВОД:
    Для POC на 8 GB: all-MiniLM остается безопасным выбором
    Для production на 16+ GB: Qwen3-Embedding идеальна!
    Для текущего проекта: можно попробовать Qwen3 и сравнить память
"""
    print(comparison)


def main():
    """Главная функция"""
    print()
    print("🔬 ТЕСТ QWEN3-EMBEDDING-0.6B-4BIT-DWQ")
    print()
    print("Цель: Проверить мультиязычную модель БЕЗ PyTorch зависимостей")
    print("Сравнение: Qwen3 vs all-MiniLM vs BGE-small")

    try:
        # Тест 1: Загрузка
        model, tokenizer = test_model_loading()

        # Тест 2: Мультиязычность
        multilingual_results = test_multilingual(model, tokenizer)

        # Тест 3: Понимание кода
        code_results = test_code_understanding(model, tokenizer)

        # Итоговое сравнение
        print_final_comparison()

        print("\n✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")

    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
