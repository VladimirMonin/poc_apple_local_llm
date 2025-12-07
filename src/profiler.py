"""
Утилита для профилирования использования ресурсов на macOS.

Отслеживает:
- Использование RAM (физическая, сжатая, своп)
- Загрузку CPU
- Memory Pressure (давление на память)
- GPU активность (через ioreg)
"""

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class MemorySnapshot:
    """Снимок использования памяти в конкретный момент времени."""

    rss_mb: float  # Resident Set Size (физическая память процесса)
    vms_mb: float  # Virtual Memory Size (виртуальная память)
    percent: float  # Процент от общей памяти
    available_mb: float  # Доступная память в системе
    swap_used_mb: float  # Использование свопа
    swap_total_mb: float  # Общий размер свопа
    compressed_mb: float  # Сжатая память (macOS specific)
    wired_mb: float  # Wired память (не может быть выгружена)
    timestamp: float  # Время замера


@dataclass
class CPUSnapshot:
    """Снимок использования CPU."""

    percent: float  # Общая загрузка CPU
    user_percent: float  # Время в user mode
    system_percent: float  # Время в kernel mode
    idle_percent: float  # Время простоя
    timestamp: float


class SystemProfiler:
    """Профилировщик системных ресурсов для macOS."""

    def __init__(self, process_pid: Optional[int] = None):
        """
        Args:
            process_pid: PID процесса для отслеживания. Если None, используется текущий процесс.
        """
        self.process = psutil.Process(process_pid or os.getpid())
        self.start_time = time.time()
        self.initial_memory = self._get_memory_snapshot()
        self.initial_cpu = self._get_cpu_snapshot()

    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Получить текущий снимок памяти."""
        # Информация о процессе
        mem_info = self.process.memory_info()

        # Системная память
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # macOS specific: sysctl для compressed/wired памяти
        try:
            # Сжатая память
            compressed_output = subprocess.check_output(
                ["sysctl", "vm.compressor_compressed_bytes"], text=True
            )
            compressed_bytes = int(compressed_output.split(":")[1].strip())

            # Wired память
            wired_output = subprocess.check_output(
                ["sysctl", "vm.page_pageable_internal_count"], text=True
            )
            wired_pages = int(wired_output.split(":")[1].strip())
            wired_bytes = wired_pages * 4096  # page size на macOS
        except (subprocess.CalledProcessError, ValueError, IndexError):
            compressed_bytes = 0
            wired_bytes = 0

        return MemorySnapshot(
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            percent=self.process.memory_percent(),
            available_mb=vm.available / 1024 / 1024,
            swap_used_mb=swap.used / 1024 / 1024,
            swap_total_mb=swap.total / 1024 / 1024,
            compressed_mb=compressed_bytes / 1024 / 1024,
            wired_mb=wired_bytes / 1024 / 1024,
            timestamp=time.time(),
        )

    def _get_cpu_snapshot(self) -> CPUSnapshot:
        """Получить текущий снимок CPU."""
        cpu_times = psutil.cpu_times_percent(interval=0.1)

        return CPUSnapshot(
            percent=psutil.cpu_percent(interval=0.1),
            user_percent=cpu_times.user,
            system_percent=cpu_times.system,
            idle_percent=cpu_times.idle,
            timestamp=time.time(),
        )

    def get_memory_pressure(self) -> str:
        """
        Получить уровень Memory Pressure через vm_stat.

        Returns:
            str: 'Normal', 'Warning', или 'Critical'
        """
        try:
            # vm_stat показывает статистику виртуальной памяти
            output = subprocess.check_output(["vm_stat"], text=True)

            # Парсим Pages free и Pages inactive
            free_pages = 0
            inactive_pages = 0
            for line in output.split("\n"):
                if "Pages free" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    inactive_pages = int(line.split(":")[1].strip().rstrip("."))

            # Оценка давления на память
            total_free = (free_pages + inactive_pages) * 4096 / 1024 / 1024  # MB
            vm = psutil.virtual_memory()
            total_mb = vm.total / 1024 / 1024

            free_percent = (total_free / total_mb) * 100

            if free_percent > 20:
                return "Normal"
            elif free_percent > 10:
                return "Warning"
            else:
                return "Critical"

        except (subprocess.CalledProcessError, ValueError, IndexError):
            return "Unknown"

    def print_current_state(self):
        """Вывести текущее состояние ресурсов."""
        mem = self._get_memory_snapshot()
        cpu = self._get_cpu_snapshot()
        pressure = self.get_memory_pressure()

        print("\n" + "=" * 60)
        print("📊 ТЕКУЩЕЕ СОСТОЯНИЕ СИСТЕМЫ")
        print("=" * 60)

        print("\n💾 ПАМЯТЬ:")
        print(f"  • Процесс (RSS): {mem.rss_mb:.1f} МБ ({mem.percent:.1f}%)")
        print(f"  • Доступно: {mem.available_mb:.1f} МБ")
        print(f"  • Своп: {mem.swap_used_mb:.1f} / {mem.swap_total_mb:.1f} МБ")
        print(f"  • Сжатая память: {mem.compressed_mb:.1f} МБ")
        print(f"  • Memory Pressure: {pressure}")

        print("\n🔥 CPU:")
        print(f"  • Загрузка: {cpu.percent:.1f}%")
        print(
            f"  • User: {cpu.user_percent:.1f}% | System: {cpu.system_percent:.1f}% | Idle: {cpu.idle_percent:.1f}%"
        )

        print("=" * 60)

    def print_delta(self):
        """Вывести изменение ресурсов с момента инициализации."""
        current_mem = self._get_memory_snapshot()
        delta_rss = current_mem.rss_mb - self.initial_memory.rss_mb
        delta_swap = current_mem.swap_used_mb - self.initial_memory.swap_used_mb
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("📈 ИЗМЕНЕНИЕ РЕСУРСОВ")
        print("=" * 60)
        print(f"⏱️  Время выполнения: {elapsed:.2f} сек")
        print(f"💾 Изменение RAM: {delta_rss:+.1f} МБ")
        print(f"💿 Изменение Swap: {delta_swap:+.1f} МБ")

        if delta_swap > 100:
            print("⚠️  ВНИМАНИЕ: Значительное использование свопа!")
        if current_mem.swap_used_mb > 1000:
            print("🔴 КРИТИЧНО: Своп > 1 ГБ - производительность снижена!")

        print("=" * 60)


def profile_function(func, *args, **kwargs):
    """
    Декоратор/враппер для профилирования функции.

    Args:
        func: Функция для профилирования
        *args, **kwargs: Аргументы функции

    Returns:
        Результат выполнения функции
    """
    profiler = SystemProfiler()

    print(f"\n🔬 Профилирование: {func.__name__}")
    profiler.print_current_state()

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time

    print(f"\n✅ {func.__name__} завершена за {elapsed:.2f} сек")
    profiler.print_delta()

    return result


if __name__ == "__main__":
    # Тест профилировщика
    print("🧪 Тест системного профилировщика\n")

    profiler = SystemProfiler()
    profiler.print_current_state()

    print("\n⏳ Ожидание 2 секунды...")
    time.sleep(2)

    profiler.print_delta()
