FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Установка Python и системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-distutils \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Запуск приложения
ENV OMP_NUM_THREADS=1
CMD ["python3", "run.py"]