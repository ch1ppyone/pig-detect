FROM nvidia/cuda:12.0.0-base-ubuntu20.04

WORKDIR /app

# Установка Python и зависимостей
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Запуск приложения
CMD ["python3", "run.py"]