FROM python:3.10

WORKDIR /app/answer_generation

# Установка зависимостей для Poetry и других нужных пакетов
RUN apt-get update && apt-get install -y curl gcc

# Установка Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Добавление Poetry в PATH
ENV PATH="/root/.local/bin:$PATH"

# Копирование конфигурационных файлов Poetry
COPY answer_generation/pyproject.toml /app/answer_generation/

# Копирование исходного кода приложения
COPY answer_generation /app/answer_generation

# Установка зависимостей проекта с новой опцией --only main
RUN poetry config virtualenvs.create false \
    && poetry install --only main

# Запуск uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug"]
