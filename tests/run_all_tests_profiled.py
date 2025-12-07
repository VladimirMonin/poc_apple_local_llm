"""
Скрипт для запуска всех тестов с детальным профилированием ресурсов.

Запускает:
1. 00_check_env.py - проверка окружения
2. 01_test_ocr.py - OCR тесты
3. 02_test_embeddings.py - Embeddings тесты
4. 03_test_whisper.py - Whisper тесты

Для каждого теста собирает:
- Время выполнения
- Использование RAM (до/после/дельта)
- Использование Swap (до/после/дельта)
- CPU load
- Memory Pressure
"""

import subprocess
import sys
import os
from pathlib import Path

# Добавляем путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiler import SystemProfiler


def run_test_with_profiling(test_name: str, test_script: str):
    """
    Запуск теста с профилированием.

    Args:
        test_name: Название теста для отчета
        test_script: Путь к скрипту теста
    """
    print("\n" + "=" * 80)
    print(f"🧪 ЗАПУСК ТЕСТА: {test_name}")
    print("=" * 80)

    # Начальное состояние системы
    profiler = SystemProfiler()
    profiler.print_current_state()

    print(f"\n▶️  Выполнение: {test_script}")
    print("-" * 80)

    # Запуск теста
    result = subprocess.run(
        [sys.executable, test_script],
        capture_output=False,  # Выводим в реальном времени
        text=True,
    )

    print("-" * 80)

    # Финальное состояние
    profiler.print_delta()

    if result.returncode == 0:
        print(f"✅ {test_name} - УСПЕШНО")
    else:
        print(f"❌ {test_name} - ОШИБКА (код: {result.returncode})")

    print("=" * 80)

    return result.returncode == 0


def main():
    """Запуск всех тестов с профилированием."""
    print("\n" + "🔬" * 40)
    print("ПОЛНОЕ ПРОФИЛИРОВАНИЕ ВСЕХ ТЕСТОВ POC")
    print("🔬" * 40)

    tests_dir = Path(__file__).parent

    tests = [
        ("Проверка окружения", "00_check_env.py"),
        ("Apple Vision OCR", "01_test_ocr.py"),
        ("MLX Embeddings", "02_test_embeddings.py"),
        ("Lightning Whisper MLX", "03_test_whisper.py"),
    ]

    results = {}

    for test_name, test_file in tests:
        test_path = tests_dir / test_file

        if not test_path.exists():
            print(f"⚠️  Пропуск {test_name}: файл не найден - {test_file}")
            results[test_name] = None
            continue

        success = run_test_with_profiling(test_name, str(test_path))
        results[test_name] = success

    # Финальный отчет
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)

    for test_name, status in results.items():
        if status is None:
            emoji = "⏭️ "
            status_text = "ПРОПУЩЕН"
        elif status:
            emoji = "✅"
            status_text = "УСПЕШНО"
        else:
            emoji = "❌"
            status_text = "ОШИБКА"

        print(f"{emoji} {test_name}: {status_text}")

    print("=" * 80)

    # Проверка общего состояния системы
    final_profiler = SystemProfiler()
    print("\n📈 ФИНАЛЬНОЕ СОСТОЯНИЕ СИСТЕМЫ:")
    final_profiler.print_current_state()

    all_passed = all(v for v in results.values() if v is not None)

    if all_passed:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        return 0
    else:
        print("\n⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
