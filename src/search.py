"""
Модуль поиска по векторным представлениям.

Использует косинусное сходство для поиска наиболее релевантных записей.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorSearch:
    """Поиск по векторным представлениям с использованием косинусного сходства."""

    @staticmethod
    def search(
        query_vector: np.ndarray,
        database_vectors: np.ndarray,
        database_metadata: List[Dict],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[Dict, float]]:
        """
        Поиск наиболее похожих записей.

        Args:
            query_vector: Вектор запроса (1D numpy array)
            database_vectors: Массив векторов базы данных (2D numpy array)
            database_metadata: Список метаданных для каждого вектора
            top_k: Количество результатов для возврата
            threshold: Минимальный порог сходства (0.0-1.0)

        Returns:
            List[(metadata, similarity_score)] отсортированный по убыванию сходства
        """
        if database_vectors is None or len(database_vectors) == 0:
            return []

        # Убедимся что query_vector 2D для sklearn
        query_vector = query_vector.reshape(1, -1)

        # Вычисление косинусного сходства
        similarities = cosine_similarity(query_vector, database_vectors)[0]

        # Создание пар (индекс, сходство)
        results = [
            (idx, score)
            for idx, score in enumerate(similarities)
            if score >= threshold
        ]

        # Сортировка по убыванию сходства
        results.sort(key=lambda x: x[1], reverse=True)

        # Ограничение top_k
        results = results[:top_k]

        # Формирование финального списка с метаданными
        final_results = [
            (database_metadata[idx], score) for idx, score in results
        ]

        return final_results

    @staticmethod
    def format_results(results: List[Tuple[Dict, float]]) -> str:
        """
        Форматирование результатов поиска для вывода.

        Args:
            results: Список результатов из search()

        Returns:
            Отформатированная строка
        """
        if not results:
            return "❌ Ничего не найдено"

        output = []
        output.append(f"\n📊 Найдено результатов: {len(results)}\n")

        for i, (metadata, score) in enumerate(results, 1):
            output.append("=" * 70)
            output.append(f"#{i} | Сходство: {score * 100:.2f}%")
            output.append("=" * 70)
            output.append(f"📅 Дата: {metadata['timestamp']}")
            output.append(f"🖼️  Скриншот: {metadata['screenshot_path']}")
            output.append(f"📝 Текст ({metadata['text_length']} символов):")
            output.append("-" * 70)

            # Обрезаем длинный текст для preview
            text = metadata['text']
            if len(text) > 300:
                text = text[:300] + "..."

            output.append(text)
            output.append("")

        return "\n".join(output)


if __name__ == "__main__":
    # Тест модуля
    print("🧪 Тест модуля поиска\n")

    # Создание тестовых данных
    # База данных: 3 вектора
    db_vectors = np.array([
        np.random.rand(384),  # Случайный вектор 1
        np.random.rand(384),  # Случайный вектор 2
        np.random.rand(384),  # Случайный вектор 3
    ])

    db_metadata = [
        {
            "id": 0,
            "timestamp": "2025-12-07T20:00:00",
            "screenshot_path": "screenshots/test1.png",
            "text": "Python - это интерпретируемый язык программирования высокого уровня.",
            "text_length": 68,
        },
        {
            "id": 1,
            "timestamp": "2025-12-07T20:05:00",
            "screenshot_path": "screenshots/test2.png",
            "text": "Machine Learning - раздел искусственного интеллекта, изучающий методы построения алгоритмов.",
            "text_length": 98,
        },
        {
            "id": 2,
            "timestamp": "2025-12-07T20:10:00",
            "screenshot_path": "screenshots/test3.png",
            "text": "Apple Silicon - семейство ARM-процессоров от Apple для компьютеров Mac.",
            "text_length": 74,
        },
    ]

    # Запрос: вектор близкий к первому элементу базы
    query = db_vectors[0] + np.random.rand(384) * 0.1  # Немного шума

    # Поиск
    print("🔍 Выполнение поиска...")
    results = VectorSearch.search(
        query_vector=query,
        database_vectors=db_vectors,
        database_metadata=db_metadata,
        top_k=3,
        threshold=0.5,
    )

    # Вывод результатов
    print(VectorSearch.format_results(results))

    print("\n✅ Тест завершен")
