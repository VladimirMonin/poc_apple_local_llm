"""
Скрипт проверки окружения для POC Apple Local LLM.

Проверяет:
- Версию Python
- Доступность MLX и Metal
- Объем доступной Unified Memory
"""

import sys
import platform


def check_python_version():
    """Проверка версии Python."""
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor in [10, 11]:
        print("✅ Версия Python корректна (3.10 или 3.11)")
        return True
    else:
        print("⚠️  Рекомендуется Python 3.10 или 3.11 для MLX")
        return False


def check_macos():
    """Проверка версии macOS."""
    if platform.system() != "Darwin":
        print("❌ Требуется macOS")
        return False
    
    mac_ver = platform.mac_ver()[0]
    print(f"macOS версия: {mac_ver}")
    
    # Ventura = 13.0+
    major_version = int(mac_ver.split('.')[0])
    if major_version >= 13:
        print("✅ macOS Ventura 13.0+ (поддержка русского OCR)")
        return True
    else:
        print("⚠️  Рекомендуется macOS Ventura 13.0+ для полной поддержки OCR")
        return False


def check_mlx():
    """Проверка установки и работы MLX."""
    try:
        import mlx.core as mx
        print(f"✅ MLX установлен: {mx.__version__}")
        
        # Проверка Metal
        if mx.metal.is_available():
            print("✅ Metal доступен")
            
            # Проверка памяти
            device = mx.default_device()
            print(f"Устройство по умолчанию: {device}")
            
            return True
        else:
            print("❌ Metal недоступен")
            return False
            
    except ImportError:
        print("❌ MLX не установлен. Запустите: pip install mlx")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке MLX: {e}")
        return False


def check_memory():
    """Проверка доступной памяти."""
    try:
        import subprocess
        
        # Получение общей памяти
        result = subprocess.run(
            ['sysctl', 'hw.memsize'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            mem_bytes = int(result.stdout.split(':')[1].strip())
            mem_gb = mem_bytes / (1024**3)
            print(f"Unified Memory: {mem_gb:.1f} ГБ")
            
            if mem_gb >= 16:
                print("✅ Достаточно памяти для комфортной работы (≥16 ГБ)")
            elif mem_gb >= 8:
                print("⚠️  8 ГБ - минимум. Рекомендуется 16 ГБ")
            else:
                print("❌ Недостаточно памяти (<8 ГБ)")
                return False
            return True
            
    except Exception as e:
        print(f"⚠️  Не удалось определить объем памяти: {e}")
        return False


def main():
    """Основная функция проверки."""
    print("=" * 60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ POC APPLE LOCAL LLM")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Проверка версии Python...")
    results.append(check_python_version())
    print()
    
    print("2. Проверка macOS...")
    results.append(check_macos())
    print()
    
    print("3. Проверка Unified Memory...")
    results.append(check_memory())
    print()
    
    print("4. Проверка MLX...")
    results.append(check_mlx())
    print()
    
    print("=" * 60)
    if all(results):
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    else:
        print("⚠️  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
    print("=" * 60)


if __name__ == "__main__":
    main()
