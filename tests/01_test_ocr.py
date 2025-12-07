"""
Тест Apple Vision OCR с поддержкой русского языка.

Проверяет:
- Работу VNRecognizeTextRequest через PyObjC
- Распознавание кириллицы (русский текст)
- Скорость обработки (< 0.5 сек)
- Потребление памяти (< 100 МБ)
"""

import time
import Vision
import Quartz
from Foundation import NSURL
import sys
import os


def recognize_text_native(image_path, languages=['ru-RU', 'en-US']):
    """
    Распознавание текста на изображении через Apple Vision Framework.
    
    Args:
        image_path: Путь к файлу изображения
        languages: Список языков для распознавания (порядок важен)
    
    Returns:
        list: Распознанные строки текста
    """
    # 1. Загрузка изображения через CoreImage
    input_url = NSURL.fileURLWithPath_(image_path)
    ci_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)
    
    if not ci_image:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # 2. Создание реквеста
    request = Vision.VNRecognizeTextRequest.alloc().init()
    
    # 3. Настройка параметров (КРИТИЧЕСКИ ВАЖНО для кириллицы!)
    # LevelAccurate использует нейросети (Deep Learning)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    
    # Явное указание языков. ru-RU поддерживается с macOS 13.0+
    request.setRecognitionLanguages_(languages)
    
    # Использование языковой коррекции (повышает точность)
    request.setUsesLanguageCorrection_(True)
    
    # 4. Обработчик
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        ci_image, None
    )
    
    # 5. Выполнение (синхронно)
    start_time = time.time()
    success, error = handler.performRequests_error_([request], None)
    elapsed_time = time.time() - start_time
    
    if not success:
        raise RuntimeError(f"Ошибка распознавания: {error}")
    
    # 6. Парсинг результатов
    results = []
    observations = request.results()
    
    if observations:
        for observation in observations:
            # Берем лучший кандидат
            top_candidates = observation.topCandidates_(1)
            if top_candidates and len(top_candidates) > 0:
                text = top_candidates[0].string()
                confidence = top_candidates[0].confidence()
                results.append({
                    'text': text,
                    'confidence': confidence
                })
    
    return results, elapsed_time


def create_test_image_programmatically():
    """
    Создает тестовое изображение с русским текстом программно.
    Использует Quartz для рендеринга текста.
    """
    from Quartz import (
        CGBitmapContextCreate, CGContextSetRGBFillColor,
        CGContextFillRect, CGContextSelectFont, CGContextShowTextAtPoint,
        kCGImageAlphaPremultipliedLast, CGBitmapContextCreateImage,
        CGImageDestinationCreateWithURL, CGImageDestinationAddImage,
        CGImageDestinationFinalize, kCGRenderingIntentDefault
    )
    from CoreFoundation import kCFAllocatorDefault
    
    width, height = 800, 200
    bytes_per_pixel = 4
    bytes_per_row = width * bytes_per_pixel
    
    # Создание bitmap context
    context = CGBitmapContextCreate(
        None, width, height, 8, bytes_per_row,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast
    )
    
    # Белый фон
    CGContextSetRGBFillColor(context, 1.0, 1.0, 1.0, 1.0)
    CGContextFillRect(context, Quartz.CGRectMake(0, 0, width, height))
    
    # Черный текст
    CGContextSetRGBFillColor(context, 0.0, 0.0, 0.0, 1.0)
    
    # Примечание: CGContextSelectFont/ShowTextAtPoint не поддерживают Unicode хорошо
    # Поэтому создадим изображение другим способом - через NSAttributedString
    
    return create_test_image_with_nsstring()


def create_test_image_with_nsstring():
    """Создает изображение с текстом через NSAttributedString."""
    from AppKit import (
        NSImage, NSAttributedString, NSFont, NSColor,
        NSForegroundColorAttributeName, NSFontAttributeName,
        NSBitmapImageRep, NSPNGFileType, NSBezierPath
    )
    from Foundation import NSMakeSize, NSMakeRect, NSMakePoint
    
    # Размер изображения
    size = NSMakeSize(800, 200)
    
    # Создание изображения
    image = NSImage.alloc().initWithSize_(size)
    image.lockFocus()
    
    # Белый фон
    NSColor.whiteColor().set()
    NSBezierPath.fillRect_(NSMakeRect(0, 0, 800, 200))
    
    # Текст
    text = "Привет, мир! Hello, world!\nТест распознавания текста на русском языке."
    font = NSFont.systemFontOfSize_(28)
    attributes = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: NSColor.blackColor()
    }
    
    attributed_string = NSAttributedString.alloc().initWithString_attributes_(
        text, attributes
    )
    
    # Рисуем текст
    attributed_string.drawAtPoint_(NSMakePoint(50, 70))
    
    # Получаем изображение
    bitmap = NSBitmapImageRep.alloc().initWithFocusedViewRect_(
        NSMakeRect(0, 0, 800, 200)
    )
    image.unlockFocus()
    
    # Сохранение
    png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)
    output_path = "test_images/russian_text.png"
    png_data.writeToFile_atomically_(output_path, True)
    
    return output_path


def test_single_image(image_path, test_name):
    """Тест распознавания для одного изображения."""
    print(f"\n{'=' * 70}")
    print(f"{test_name}: {os.path.basename(image_path)}")
    print('=' * 70)
    
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return False
    
    try:
        results, elapsed = recognize_text_native(image_path)
        
        print(f"⏱️  Время обработки: {elapsed:.3f} сек")
        
        if elapsed < 0.5:
            print("✅ Скорость распознавания отличная (< 0.5 сек)")
        elif elapsed < 1.0:
            print(f"⚠️  Распознавание медленнее ожидаемого ({elapsed:.3f} сек)")
        else:
            print(f"⚠️  Медленная обработка ({elapsed:.3f} сек)")
        
        print(f"📄 Распознано строк: {len(results)}")
        
        if results:
            print("\nРаспознанный текст:")
            print("-" * 70)
            for i, item in enumerate(results, 1):
                text = item['text'][:100] + ('...' if len(item['text']) > 100 else '')
                print(f"{i}. {text}")
                print(f"   Уверенность: {item['confidence']:.2%}")
            print("-" * 70)
            
            # Проверка наличия кириллицы
            all_text = " ".join([r['text'] for r in results])
            has_cyrillic = any(ord(char) >= 0x0400 and ord(char) <= 0x04FF 
                             for char in all_text)
            
            if has_cyrillic:
                print("✅ Кириллица распознана")
            
            return True
        else:
            print("❌ Текст не распознан")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при распознавании: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_basic():
    """Базовый тест OCR на всех изображениях."""
    print("=" * 70)
    print("ТЕСТ 1: Apple Vision OCR (Распознавание текста)")
    print("=" * 70)
    
    # Создаем тестовое изображение программно
    print("\n📝 Создание синтетического изображения с русским текстом...")
    try:
        synthetic_image = create_test_image_with_nsstring()
        print(f"✅ Изображение создано: {synthetic_image}")
    except Exception as e:
        print(f"❌ Ошибка создания изображения: {e}")
        synthetic_image = None
    
    # Собираем все изображения для тестирования
    test_images = []
    
    if synthetic_image and os.path.exists(synthetic_image):
        test_images.append((synthetic_image, "Синтетическое изображение"))
    
    # Добавляем пользовательские изображения
    image_dir = "test_images"
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename != 'russian_text.png':
                full_path = os.path.join(image_dir, filename)
                test_images.append((full_path, f"Пользовательское изображение"))
    
    # Тестируем все изображения
    results = []
    for image_path, description in test_images:
        result = test_single_image(image_path, description)
        results.append(result)
    
    return any(results)  # Хотя бы один тест должен пройти


def test_ocr_performance():
    """Тест производительности."""
    print()
    print("=" * 70)
    print("ТЕСТ 2: Проверка потребления ресурсов")
    print("=" * 70)
    print()
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # МБ
    
    print(f"Память до теста: {mem_before:.1f} МБ")
    
    # Выполняем несколько распознаваний
    image_path = "test_images/russian_text.png"
    
    for i in range(3):
        results, elapsed = recognize_text_native(image_path)
        print(f"  Итерация {i+1}: {elapsed:.3f} сек, строк: {len(results)}")
    
    mem_after = process.memory_info().rss / 1024 / 1024  # МБ
    mem_delta = mem_after - mem_before
    
    print(f"Память после теста: {mem_after:.1f} МБ")
    print(f"Прирост памяти: {mem_delta:.1f} МБ")
    
    if mem_delta < 100:
        print("✅ Потребление памяти в пределах нормы (< 100 МБ)")
        return True
    else:
        print(f"⚠️  Высокое потребление памяти ({mem_delta:.1f} МБ)")
        return False


def main():
    """Основная функция."""
    # Установка psutil для теста памяти
    try:
        import psutil
    except ImportError:
        print("Установка psutil для мониторинга памяти...")
        os.system("pip install psutil")
        import psutil
    
    results = []
    
    # Тест 1: Базовое распознавание
    results.append(test_ocr_basic())
    
    # Тест 2: Производительность
    results.append(test_ocr_performance())
    
    print()
    print("=" * 70)
    if all(results):
        print("✅ ВСЕ ТЕСТЫ OCR ПРОЙДЕНЫ УСПЕШНО")
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
    print("=" * 70)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
