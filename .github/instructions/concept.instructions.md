---
applyTo: "**"
name: "ConceptInstructions"
description: "Концепция POC Local-LLM MacOS"
---

## Библиотеки проекта и их Context7 ID

Если ты не уверен, или видишь что неполучается так как хочешь сделать, изучай документацию через Context7 по указанным ID. Мы специально их подготовили, чтобы ты не тратил время на поиск идентификатора конкретной библиотеки.

В данном проекте используются следующие ключевые библиотеки для построения мультимодального AI-стека на Apple Silicon:

| Библиотека                | Context7 ID                              | Описание                                                      | Код примеров | Репутация | Score |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------- | ------------ | --------- | ----- |
| **MLX**                   | `/ml-explore/mlx`                        | Фреймворк для ML на Apple Silicon с поддержкой Unified Memory | 321          | Medium    | 88.5  |
| **MLX-LM**                | `/ml-explore/mlx-lm`                     | Пакет для работы с LLM на Apple Silicon через MLX             | 68           | Medium    | 71.3  |
| **MLX-VLM**               | `/blaizzy/mlx-vlm`                       | Пакет для Vision-Language моделей на MLX                      | 122          | High      | —     |
| **MLX Embeddings**        | `/blaizzy/mlx-embeddings`                | Локальные эмбеддинги на Mac через MLX                         | 11           | High      | 48.0  |
| **Lightning Whisper MLX** | `/mustafaaljadery/lightning-whisper-mlx` | Быстрая реализация Whisper для Apple Silicon                  | 14           | High      | 75.9  |
| **PyObjC**                | `/ronaldoussoren/pyobjc`                 | Python <-> Objective-C мост для macOS фреймворков             | 876          | High      | —     |
| **NumPy**                 | `/numpy/numpy`                           | Базовая библиотека для научных вычислений                     | 2157         | Unknown   | 79.9  |
| **PyTorch + Torchvision** | `/pytorch/pytorch`                       | Нужны для препроцессора Qwen3-VL (AutoVideoProcessor)         | 2516         | High      | 91.0  |
| **Sentence Transformers** | `/huggingface/sentence-transformers`     | Фреймворк для эмбеддингов (альтернатива MLX)                  | 539          | High      | 94.3  |

### Рекомендации по использованию:

1. **Приоритет MLX**: Для всех операций на Apple Silicon используйте MLX-версии библиотек вместо PyTorch
2. **Native OCR**: PyObjC обеспечивает доступ к Apple Vision Framework без накладных расходов
3. **Эмбеддинги**: MLX Embeddings предпочтительнее sentence-transformers для избежания проблем с MPS backend
4. **Whisper**: Lightning Whisper MLX оптимизирован для Apple Silicon и работает быстрее стандартной реализации
5. **Qwen3-VL**: Использует MLX для инференса, но требует PyTorch/Torchvision для препроцессинга (гибридный подход)

### Используемые модели:

- **Vision-Language**: `mlx-community/Qwen3-VL-4B-Instruct-4bit` (~3.3 ГБ, 4B параметров)
- **Embeddings**: `mlx-community/all-MiniLM-L6-v2-4bit` (384-dim, ~150 МБ)
- **Speech-to-Text**: `mlx-community/whisper-large-v3-turbo` (~800 МБ)

### Критические зависимости:

- **Python**: 3.10 или 3.11 (максимальная стабильность с MLX)
- **macOS**: Ventura (13.0+) для поддержки русского языка в Vision OCR
- **Память**: Рекомендуется 16 ГБ RAM для стабильной работы полного стека
- **Дополнительно**: torchvision (для Qwen3-VL AutoVideoProcessor - обработка изображений/видео)
