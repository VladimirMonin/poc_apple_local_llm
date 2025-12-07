"""
Модуль хранилища для векторов и метаданных скриншотов.

Использует простое JSON-хранилище для POC.
Векторы сохраняются отдельно в numpy формате для эффективности.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class VectorStorage:
    """Простое хранилище для векторов и метаданных скриншотов."""

    def __init__(self, storage_dir: Path):
        """
        Args:
            storage_dir: Директория для хранения данных.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.storage_dir / "metadata.json"
        self.vectors_file = self.storage_dir / "vectors.npy"

        # Загрузка существующих данных
        self.metadata = self._load_metadata()
        self.vectors = self._load_vectors()

    def _load_metadata(self) -> List[Dict]:
        """Загрузить метаданные из JSON."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        """Сохранить метаданные в JSON."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load_vectors(self) -> Optional[np.ndarray]:
        """Загрузить векторы из numpy файла."""
        if self.vectors_file.exists():
            return np.load(str(self.vectors_file))
        return None

    def _save_vectors(self):
        """Сохранить векторы в numpy файл."""
        if self.vectors is not None:
            np.save(str(self.vectors_file), self.vectors)

    def add(
        self,
        vector: np.ndarray,
        text: str,
        screenshot_path: str,
        timestamp: Optional[str] = None,
    ) -> int:
        """
        Добавить новую запись в хранилище.

        Args:
            vector: Вектор эмбеддинга (numpy array)
            text: Распознанный текст
            screenshot_path: Путь к скриншоту
            timestamp: Временная метка (если None, генерируется автоматически)

        Returns:
            int: ID добавленной записи
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Добавление метаданных
        record_id = len(self.metadata)
        metadata_entry = {
            "id": record_id,
            "timestamp": timestamp,
            "screenshot_path": str(screenshot_path),
            "text": text,
            "text_length": len(text),
        }
        self.metadata.append(metadata_entry)

        # Добавление вектора
        vector = vector.reshape(1, -1)  # Ensure 2D shape
        if self.vectors is None:
            self.vectors = vector
        else:
            self.vectors = np.vstack([self.vectors, vector])

        # Сохранение
        self._save_metadata()
        self._save_vectors()

        print(f"✅ Добавлена запись #{record_id}")
        print(f"   📝 Текст: {len(text)} символов")
        print(f"   🔢 Вектор: {vector.shape}")

        return record_id

    def get(self, record_id: int) -> Optional[Dict]:
        """
        Получить запись по ID.

        Args:
            record_id: ID записи

        Returns:
            Dict с метаданными и вектором или None
        """
        if 0 <= record_id < len(self.metadata):
            metadata = self.metadata[record_id]
            vector = self.vectors[record_id] if self.vectors is not None else None
            return {**metadata, "vector": vector}
        return None

    def get_all_vectors(self) -> Optional[np.ndarray]:
        """Получить все векторы."""
        return self.vectors

    def get_all_metadata(self) -> List[Dict]:
        """Получить все метаданные."""
        return self.metadata

    def count(self) -> int:
        """Получить количество записей."""
        return len(self.metadata)

    def clear(self):
        """Очистить все данные (ОПАСНО!)."""
        self.metadata = []
        self.vectors = None
        self._save_metadata()
        if self.vectors_file.exists():
            self.vectors_file.unlink()
        print("⚠️  Хранилище очищено")


if __name__ == "__main__":
    # Тест модуля
    print("🧪 Тест модуля хранилища\n")

    # Создание тестового хранилища
    storage = VectorStorage(Path("test_storage"))

    # Создание тестовых данных
    test_vector = np.random.rand(384)  # 384-размерный вектор (как в all-MiniLM)
    test_text = "Тестовый текст для проверки хранилища"
    test_screenshot = "test_images/screenshot_test.png"

    # Добавление записи
    record_id = storage.add(
        vector=test_vector, text=test_text, screenshot_path=test_screenshot
    )

    # Получение записи
    record = storage.get(record_id)
    print(f"\n📄 Получена запись #{record_id}:")
    print(f"   Timestamp: {record['timestamp']}")
    print(f"   Screenshot: {record['screenshot_path']}")
    print(f"   Text length: {record['text_length']}")
    print(f"   Vector shape: {record['vector'].shape}")

    # Статистика
    print(f"\n📊 Статистика хранилища:")
    print(f"   Всего записей: {storage.count()}")
    print(
        f"   Векторов: {storage.get_all_vectors().shape if storage.get_all_vectors() is not None else 'None'}"
    )

    # Очистка тестового хранилища
    storage.clear()
    print(f"\n✅ Тест завершен")
