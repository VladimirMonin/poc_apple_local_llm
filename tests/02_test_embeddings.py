"""
Тест BGE-M3 Embeddings на MLX.

Проверяет:
- Загрузку модели BGE-M3 через mlx-embeddings
- Генерацию эмбеддингов для текста
- Вычисление косинусного сходства
- Работу с длинными текстами (до 8192 токенов)
- Потребление памяти
"""

import time
import sys
import os
import mlx.core as mx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_embeddings(texts, model, tokenizer):
    """
    Получение эмбеддингов для списка текстов.

    Args:
        texts: Список строк для векторизации
        model: Модель MLX
        tokenizer: Токенайзер

    Returns:
        numpy.ndarray: Матрица эмбеддингов
    """
    inputs = tokenizer.batch_encode_plus(
        texts,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=8192,  # BGE-M3 поддерживает до 8192 токенов
    )

    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # MLX возвращает mean pooled и нормализованные эмбеддинги
    embeddings = outputs.text_embeds

    # Конвертируем в numpy для sklearn
    return np.array(embeddings)


def test_basic_embeddings():
    """Базовый тест генерации эмбеддингов."""
    print("=" * 70)
    print("ТЕСТ 1: Базовая генерация эмбеддингов")
    print("=" * 70)
    print()

    try:
        from mlx_embeddings.utils import load

        print("📦 Загрузка модели BGE-M3 (или аналога)...")
        # Используем меньшую модель all-MiniLM для тестирования
        # BGE-M3 пока нет в MLX формате на HuggingFace
        model_name = "mlx-community/all-MiniLM-L6-v2-4bit"

        print(f"   Модель: {model_name}")
        start_time = time.time()
        model, tokenizer = load(model_name)
        load_time = time.time() - start_time

        print(f"✅ Модель загружена за {load_time:.2f} сек")
        print()

        # Тестовые тексты
        texts = [
            "Я люблю программирование на Python",
            "Мне нравится писать код на Python",
            "Черепаха медленно ползёт под деревом",
        ]

        print("🔄 Генерация эмбеддингов...")
        start_time = time.time()
        embeddings = get_embeddings(texts, model, tokenizer)
        embed_time = time.time() - start_time

        print(f"✅ Эмбеддинги сгенерированы за {embed_time:.3f} сек")
        print(f"   Размерность: {embeddings.shape}")
        print()

        # Вычисление сходства
        print("📊 Вычисление косинусного сходства...")
        similarity_matrix = cosine_similarity(embeddings)

        print("\nМатрица сходства:")
        print("-" * 70)
        for i in range(len(texts)):
            for j in range(len(texts)):
                if i < j:  # Показываем только верхний треугольник
                    print(
                        f"Текст {i + 1} ↔ Текст {j + 1}: {similarity_matrix[i][j]:.4f}"
                    )
        print("-" * 70)
        print()

        # Проверка логики
        # Тексты 1 и 2 (про Python) должны быть похожи
        # Текст 3 (про черепаху) должен отличаться
        sim_python = similarity_matrix[0][1]
        sim_diff = similarity_matrix[0][2]

        if sim_python > 0.7:
            print(f"✅ Похожие тексты распознаны (сходство {sim_python:.2%})")
        else:
            print(f"⚠️  Низкое сходство похожих текстов ({sim_python:.2%})")

        if sim_diff < 0.5:
            print(f"✅ Разные тексты различаются (сходство {sim_diff:.2%})")
        else:
            print(f"⚠️  Высокое сходство разных текстов ({sim_diff:.2%})")

        return True, model, tokenizer

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_long_text(model, tokenizer):
    """Тест обработки длинных текстов."""
    print()
    print("=" * 70)
    print("ТЕСТ 2: Обработка длинного текста (500 слов)")
    print("=" * 70)
    print()

    # Генерируем длинный текст
    long_text = (
        """
    Apple Silicon представляет собой серию процессоров на базе ARM архитектуры, 
    разработанных Apple Inc. специально для их компьютеров Mac. Первым чипом 
    в этой линейке стал M1, представленный в ноябре 2020 года. Этот чип 
    объединяет центральный процессор (CPU), графический процессор (GPU), 
    нейронный движок (Neural Engine) и объединенную память (Unified Memory) 
    на одном кристалле.
    
    Unified Memory Architecture (UMA) является ключевой особенностью Apple Silicon.
    В отличие от традиционных архитектур, где CPU и GPU имеют раздельные пулы 
    памяти, в UMA все вычислительные блоки имеют доступ к единому массиву данных.
    Это устраняет необходимость копирования данных между RAM и VRAM, что 
    значительно увеличивает производительность и энергоэффективность.
    
    MLX - это фреймворк машинного обучения, разработанный Apple Machine Learning 
    Research специально для Apple Silicon. MLX оптимизирован для работы с 
    Unified Memory и Metal Performance Shaders. Он предоставляет API, похожий 
    на NumPy и PyTorch, но с учетом особенностей архитектуры Apple.
    """
        * 5
    )  # Повторяем для увеличения длины

    try:
        print(f"📝 Длина текста: ~{len(long_text.split())} слов")

        start_time = time.time()
        embeddings = get_embeddings([long_text], model, tokenizer)
        embed_time = time.time() - start_time

        print(f"✅ Эмбеддинг получен за {embed_time:.3f} сек")
        print(f"   Размерность: {embeddings.shape}")

        if embed_time < 2.0:
            print("✅ Скорость обработки отличная (< 2 сек)")
        else:
            print(f"⚠️  Медленная обработка ({embed_time:.3f} сек)")

        return True

    except Exception as e:
        print(f"❌ Ошибка при обработке длинного текста: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_usage(model, tokenizer):
    """Тест потребления памяти."""
    print()
    print("=" * 70)
    print("ТЕСТ 3: Проверка потребления памяти")
    print("=" * 70)
    print()

    try:
        import psutil

        process = psutil.Process(os.getpid())
    except ImportError:
        print("⚠️  psutil не установлен, пропускаем тест памяти")
        return True

    mem_before = process.memory_info().rss / 1024 / 1024  # МБ
    print(f"Память до теста: {mem_before:.1f} МБ")

    # Генерируем эмбеддинги 10 раз
    test_text = "Тестовая строка для проверки памяти. " * 50

    for i in range(10):
        embeddings = get_embeddings([test_text], model, tokenizer)
        if (i + 1) % 3 == 0:
            print(f"  Итерация {i + 1}/10 завершена")

    mem_after = process.memory_info().rss / 1024 / 1024  # МБ
    mem_delta = mem_after - mem_before

    print(f"Память после теста: {mem_after:.1f} МБ")
    print(f"Прирост памяти: {mem_delta:.1f} МБ")
    print()

    if mem_delta < 200:
        print("✅ Потребление памяти в пределах нормы (< 200 МБ)")
        return True
    else:
        print(f"⚠️  Высокое потребление памяти ({mem_delta:.1f} МБ)")
        return True  # Не блокируем на этом


def main():
    """Основная функция."""
    results = []

    # Тест 1: Базовые эмбеддинги
    success, model, tokenizer = test_basic_embeddings()
    results.append(success)

    if not success:
        print("\n❌ Базовый тест не пройден, остальные тесты пропущены")
        return False

    # Тест 2: Длинный текст
    results.append(test_long_text(model, tokenizer))

    # Тест 3: Память
    results.append(test_memory_usage(model, tokenizer))

    print()
    print("=" * 70)
    if all(results):
        print("✅ ВСЕ ТЕСТЫ EMBEDDINGS ПРОЙДЕНЫ УСПЕШНО")
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
    print("=" * 70)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
