# POC Apple Local LLM

Локальный мультимодальный AI-стек на Apple Silicon (MLX)

## Требования

- macOS Ventura 13.0+
- Python 3.10 или 3.11
- Apple Silicon (M1/M2/M3/M4)
- Рекомендуется: 16 ГБ RAM

## Установка

```bash
# Создание виртуального окружения
python3.11 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

## Структура проекта

```
poc_apple_local_llm/
├── tests/              # Модульные тесты компонентов
├── src/                # Основной код
├── .github/            # Инструкции и документация
└── requirements.txt    # Зависимости
```

## Документация

См. `.github/instructions/` для подробной информации о концепции и библиотеках.
