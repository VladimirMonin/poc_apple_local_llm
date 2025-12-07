"""
Lightweight Core - "Вторая память".

Интегрирует:
- Screenshot capture (Quartz)
- OCR (Apple Vision)
- Embeddings (MLX)
- Vector Storage
- Semantic Search

Без тяжелых LLM - только базовые компоненты для POC.
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from screenshot import capture_screenshot
from storage import VectorStorage
from search import VectorSearch
import Vision
import Quartz
from Foundation import NSURL
import mlx.core as mx
from mlx_embeddings.utils import load
import numpy as np


class LightweightCore:
    """Легковесное ядро для запоминания и поиска скриншотов."""

    def __init__(self, storage_dir: Path = None):
        """
        Args:
            storage_dir: Директория для хранения данных.
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "memory_storage"

        self.storage = VectorStorage(storage_dir)

        # Ленивая загрузка моделей (загружаем только при использовании)
        self._embedding_model = None
        self._tokenizer = None

        print("✅ Lightweight Core инициализирован")
        print(f"   📁 Хранилище: {storage_dir}")
        print(f"   📊 Записей в базе: {self.storage.count()}")

    def _get_embedding_model(self):
        """Ленивая загрузка модели эмбеддингов."""
        if self._embedding_model is None:
            print("\n📦 Загрузка модели эмбеддингов...")
            self._embedding_model, self._tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")
            print("✅ Модель загружена")
        return self._embedding_model, self._tokenizer

    def _ocr_recognize(self, image_path: str) -> str:
        """
        Распознавание текста через Apple Vision.

        Args:
            image_path: Путь к изображению

        Returns:
            Распознанный текст
        """
        # Загрузка изображения
        input_url = NSURL.fileURLWithPath_(image_path)
        ci_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)

        if not ci_image:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Создание реквеста OCR
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(["ru-RU", "en-US"])
        request.setUsesLanguageCorrection_(True)

        # Обработчик
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None
        )

        # Выполнение
        success = handler.performRequests_error_([request], None)

        if not success:
            return ""

        # Сбор результатов
        observations = request.results()
        if not observations:
            return ""

        texts = [obs.text() for obs in observations]
        return "\n".join(texts)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Генерация эмбеддинга для текста.

        Args:
            text: Текст для векторизации

        Returns:
            Numpy array с эмбеддингом
        """
        model, tokenizer = self._get_embedding_model()
        
        # Токенизация
        inputs = tokenizer.batch_encode_plus(
            [text],
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,  # Ограничиваем для экономии памяти
        )
        
        # Генерация эмбеддинга
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embeddings = outputs.text_embeds
        
        return np.array(embeddings[0])

    def remember(self, screenshot_path: str = None) -> int:
        """
        Запомнить скриншот (захватить + OCR + embedding + сохранить).

        Args:
            screenshot_path: Путь к существующему скриншоту.
                           Если None, делается новый скриншот.

        Returns:
            ID сохраненной записи
        """
        print("\n" + "=" * 70)
        print("📸 ЗАПОМИНАНИЕ")
        print("=" * 70)

        # Шаг 1: Получение скриншота
        if screenshot_path is None:
            print("\n1️⃣  Захват скриншота...")
            screenshot_path = capture_screenshot()
        else:
            print(f"\n1️⃣  Использование существующего скриншота: {screenshot_path}")

        # Шаг 2: OCR
        print("\n2️⃣  Распознавание текста (OCR)...")
        text = self._ocr_recognize(str(screenshot_path))
        print(f"   ✅ Распознано: {len(text)} символов")
        if len(text) > 100:
            print(f"   Preview: {text[:100]}...")
        else:
            print(f"   Текст: {text}")

        if not text.strip():
            print("   ⚠️  Текст не найден на изображении")
            return -1

        # Шаг 3: Генерация эмбеддинга
        print("\n3️⃣  Генерация эмбеддинга...")
        vector = self._generate_embedding(text)
        print(f"   ✅ Вектор создан: размерность {vector.shape}")

        # Шаг 4: Сохранение
        print("\n4️⃣  Сохранение в базу...")
        record_id = self.storage.add(
            vector=vector, text=text, screenshot_path=str(screenshot_path)
        )

        print("\n" + "=" * 70)
        print(f"✅ ЗАПОМНЕНО! Record ID: {record_id}")
        print("=" * 70)

        return record_id

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Поиск похожих скриншотов по текстовому запросу.

        Args:
            query: Текстовый запрос
            top_k: Количество результатов

        Returns:
            List[(metadata, similarity_score)]
        """
        print("\n" + "=" * 70)
        print(f"🔍 ПОИСК: \"{query}\"")
        print("=" * 70)

        if self.storage.count() == 0:
            print("\n❌ База данных пуста. Сначала запомните несколько скриншотов.")
            return []

        # Генерация эмбеддинга запроса
        print("\n1️⃣  Векторизация запроса...")
        query_vector = self._generate_embedding(query)

        # Поиск
        print("\n2️⃣  Поиск в базе...")
        results = VectorSearch.search(
            query_vector=query_vector,
            database_vectors=self.storage.get_all_vectors(),
            database_metadata=self.storage.get_all_metadata(),
            top_k=top_k,
            threshold=0.3,  # Минимальное сходство 30%
        )

        # Вывод результатов
        print(VectorSearch.format_results(results))

        return results


if __name__ == "__main__":
    # Демонстрация работы
    print("\n" + "🧪" * 35)
    print("ДЕМОНСТРАЦИЯ LIGHTWEIGHT CORE")
    print("🧪" * 35)

    # Инициализация
    core = LightweightCore()

    # Тест 1: Запоминание существующих скриншотов из test_images
    test_images = Path(__file__).parent.parent / "test_images"
    if test_images.exists():
        images = list(test_images.glob("*.png")) + list(test_images.glob("*.jpg"))
        print(f"\n📁 Найдено {len(images)} тестовых изображений")

        for img in images[:2]:  # Берем первые 2 для демо
            core.remember(str(img))

    # Тест 2: Поиск
    if core.storage.count() > 0:
        # Поиск по разным запросам
        test_queries = [
            "Python",
            "GLM API",
            "ошибка",
        ]

        for query in test_queries:
            core.search(query, top_k=2)

    print("\n✅ Демонстрация завершена")
