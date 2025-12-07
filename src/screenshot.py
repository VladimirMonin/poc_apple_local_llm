"""
Модуль для захвата скриншотов экрана на macOS.

Использует нативный фреймворк Quartz для захвата экрана без внешних зависимостей.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import Quartz
from Quartz import (
    CGWindowListCreateImage,
    CGRectInfinite,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowImageDefault,
)
from AppKit import NSBitmapImageRep, NSPNGFileType


def capture_screenshot(save_path: Optional[Path] = None) -> Path:
    """
    Захват скриншота текущего экрана.

    Args:
        save_path: Путь для сохранения скриншота. Если None, создается автоматически.

    Returns:
        Path: Путь к сохраненному файлу скриншота.

    Raises:
        RuntimeError: Если захват экрана не удался.
    """
    # Захват экрана через Quartz (нативный API macOS)
    image = CGWindowListCreateImage(
        CGRectInfinite,
        kCGWindowListOptionOnScreenOnly,
        0,  # windowID=0 означает весь экран
        kCGWindowImageDefault,
    )

    if image is None:
        raise RuntimeError("Не удалось захватить скриншот")

    # Конвертация CGImage в NSBitmapImageRep для сохранения
    bitmap = NSBitmapImageRep.alloc().initWithCGImage_(image)

    if bitmap is None:
        raise RuntimeError("Не удалось конвертировать изображение")

    # Генерация имени файла, если не указан путь
    if save_path is None:
        screenshots_dir = Path(__file__).parent.parent / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = screenshots_dir / f"screenshot_{timestamp}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохранение в PNG
    png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)
    png_data.writeToFile_atomically_(str(save_path), True)

    print(f"📸 Скриншот сохранен: {save_path}")
    return save_path


if __name__ == "__main__":
    # Тест модуля
    print("🧪 Тест модуля захвата скриншотов")
    screenshot_path = capture_screenshot()
    print(f"✅ Скриншот создан: {screenshot_path}")
    print(f"📏 Размер файла: {screenshot_path.stat().st_size / 1024:.1f} КБ")
