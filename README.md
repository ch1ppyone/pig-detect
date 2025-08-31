# 🐷 Pig Detection & Monitoring System

Интеллектуальная система детекции и мониторинга поведения свиней на основе машинного обучения с веб-интерфейсом в реальном времени.

## 📋 Содержание

- [Описание](#описание)
- [Возможности](#возможности)
- [Технологии](#технологии)
- [Архитектура](#архитектура)
- [Установка](#установка)
- [Запуск](#запуск)
- [Обучение моделей](#обучение-моделей)
- [API документация](#api-документация)
- [Docker](#docker)
- [Конфигурация](#конфигурация)
- [Troubleshooting](#troubleshooting)

## 🎯 Описание

Pig Detection & Monitoring System - это комплексное решение для автоматического анализа поведения свиней в реальном времени. Система использует современные алгоритмы компьютерного зрения для детекции животных и классификации их состояний.

### 🔍 Основные состояния классификации:
- **Feeding** - Кормление
- **Lateral_lying** - Лежание на боку  
- **Sitting** - Сидение
- **Standing** - Стояние
- **Sternal_lying** - Лежание на груди

## ✨ Возможности

### 🤖 Машинное обучение
- **YOLO11** детекция объектов с трекингом ByteTrack
- **MobileNetV3** классификация поведенческих состояний
- Поддержка GPU (CUDA) и CPU режимов
- Настраиваемые пороги уверенности и фильтры

### 🌐 Веб-интерфейс
- Современный responsive UI на Bootstrap 5
- Видеопоток в реальном времени через WebSocket
- Интерактивная панель управления с KPI
- Система аутентификации с ролями
- Многоязычная поддержка (RU/EN)

### 📊 Аналитика и мониторинг
- Статистика детекций и классификаций
- Логирование событий в реальном времени
- Экспорт данных и отчетов
- Графики и диаграммы активности

### 🎥 Источники видео
- Веб-камеры (USB/встроенные)
- Видеофайлы (MP4, AVI, MOV, MKV)
- RTSP потоки IP-камер
- Статические изображения

## 🛠 Технологии

### Backend
```
Python 3.10+
Flask + Flask-SocketIO (веб-сервер)
PyTorch + Ultralytics YOLO (ML)
OpenCV (обработка видео)
SQLite (база данных)
```

### Frontend
```
HTML5 + CSS3 + JavaScript
Bootstrap 5 (UI фреймворк)
WebSocket (real-time коммуникация)
Chart.js (графики)
```

### ML/AI
```
YOLO11 (детекция объектов)
ByteTrack (трекинг объектов)
MobileNetV3 (классификация)
Albumentations (аугментация данных)
```

## 🏗 Архитектура

```
pig-detect/
├── 📁 app/                     # Основное приложение
│   ├── __init__.py            # Flask app factory
│   ├── auth.py                # Аутентификация
│   ├── config.py              # Конфигурация
│   ├── db.py                  # База данных
│   ├── detector.py            # YOLO детектор
│   ├── logging.py             # Логирование
│   ├── routes.py              # API маршруты
│   └── video_processor.py     # Обработка видео
├── 📁 static/                 # Статические файлы
│   ├── css/styles.css         # Стили
│   ├── js/main.js             # Frontend логика
│   └── icons/                 # Иконки
├── 📁 template/               # HTML шаблоны
│   ├── index.html             # Главная страница
│   ├── login.html             # Авторизация
│   └── preview.html           # Предпросмотр
├── 📁 models/                 # ML модели
│   ├── yolo11n-mod.pt         # YOLO модель
│   ├── pig_model.pth          # PyTorch классификатор
│   └── bytetrack.yaml         # Конфиг трекера
├── 📁 train/                  # Обучение моделей
│   ├── train_detect.py        # Обучение детекции
│   ├── train_classify.py      # Обучение классификации
│   ├── dataset_augment.py     # Аугментация данных
│   ├── crop_objects.py        # Кроп объектов
│   └── dataset/               # Датасеты
└── 📁 uploads/                # Загруженные файлы
```

## 🚀 Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd pig-detect
```

### 2. Создание виртуального окружения
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Настройка конфигурации
```bash
# Создание .env файла (опционально)
cp config.env.example .env
```

### 5. Подготовка моделей
Убедитесь, что модели находятся в папке `models/`:
- `yolo11n-mod.pt` - YOLO модель детекции
- `pig_model.pth` - PyTorch классификатор
- `bytetrack.yaml` - конфигурация трекера

## 🎬 Запуск

### Быстрый запуск
```bash
python run.py
```

### Запуск с параметрами
```bash
# Указание видеофайла
python run.py --video path/to/video.mp4

# Использование веб-камеры
python run.py --video webcam

# Настройка устройства
python run.py --device cuda  # или cpu

# Настройка модели
python run.py --model models/custom_model.pt
```

### Доступ к системе
- **URL**: http://localhost:9001
- **Логин**: admin
- **Пароль**: admin

## 🏋️ Обучение моделей

### 1. Подготовка датасета

#### Структура датасета для детекции:
```
dataset/
├── train/
│   ├── images/     # Изображения
│   └── labels/     # YOLO разметка (.txt)
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml    # Конфигурация датасета
```

#### Структура датасета для классификации:
```
dataset/
├── Feeding/        # Изображения кормления
├── Lateral_lying/  # Изображения лежания на боку
├── Sitting/        # Изображения сидения
├── Standing/       # Изображения стояния
└── Sternal_lying/  # Изображения лежания на груди
```

### 2. Аугментация данных

#### Быстрая аугментация датасета:
```bash
python train/dataset_augment.py \
    --source train/object/pig \
    --output train/object/pig_augmented \
    --count 3 \
    --workers 8
```

**Параметры:**
- `--source` - исходный датасет
- `--output` - выходной датасет
- `--count` - количество аугментаций на изображение
- `--workers` - количество потоков

### 3. Кроп объектов из детекций

```bash
python train/crop_objects.py \
    train/object/pig \
    train/dataset/cropped \
    0
```

**Параметры:**
- Входная папка с датасетом YOLO
- Выходная папка для кропов
- ID класса для кропа (0 = pig)

### 4. Обучение детекции (YOLO)

#### Новое обучение:
```bash
python train/train_detect.py \
    --dataset_dir train/object/pig_augmented \
    --epochs 100 \
    --img_size 640 \
    --batch_size 16
```

#### Продолжение обучения:
```bash
python train/train_detect.py \
    --dataset_dir train/object/pig_augmented \
    --weights runs/detect/train/weights/last.pt \
    --resume \
    --epochs 50
```

#### С предобученными весами:
```bash
python train/train_detect.py \
    --dataset_dir train/object/pig_augmented \
    --weights yolo11n.pt \
    --epochs 100
```

**Параметры:**
- `--dataset_dir` - путь к датасету
- `--epochs` - количество эпох
- `--img_size` - размер изображения
- `--batch_size` - размер батча
- `--weights` - путь к весам
- `--resume` - продолжить обучение

### 5. Обучение классификации (MobileNetV3)

#### Базовое обучение:
```bash
python train/train_classify.py \
    --dataset train/dataset \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

#### С сильной аугментацией:
```bash
python train/train_classify.py \
    --dataset train/dataset \
    --epochs 100 \
    --batch-size 16 \
    --heavy-aug \
    --device cuda \
    --out models/pig_model_heavy.pth
```

**Параметры:**
- `--dataset` - путь к датасету ImageFolder
- `--epochs` - количество эпох
- `--batch-size` - размер батча
- `--device` - устройство (cuda/cpu)
- `--imgsz` - размер изображения
- `--heavy-aug` - сильная аугментация
- `--out` - файл для сохранения модели

### 6. Полный пайплайн обучения

```bash
# 1. Аугментация исходного датасета
python train/dataset_augment.py \
    --source train/object/pig \
    --output train/object/pig_aug \
    --count 3

# 2. Обучение детекции
python train/train_detect.py \
    --dataset_dir train/object/pig_aug \
    --epochs 100

# 3. Кроп объектов для классификации
python train/crop_objects.py \
    train/object/pig \
    train/dataset/cropped

# 4. Обучение классификации
python train/train_classify.py \
    --dataset train/dataset \
    --epochs 50 \
    --heavy-aug
```

## 📡 API документация

### WebSocket события

#### Подключение к видеопотоку:
```javascript
socket.emit('start_video', {
    video_path: 'webcam',  // или путь к файлу
    conf: 0.5
});
```

#### Получение кадров:
```javascript
socket.on('video_frame', function(data) {
    // data.data содержит base64 изображение
});
```

#### Получение логов:
```javascript
socket.on('detection_log', function(data) {
    // data.logs - массив логов детекций
});
```

### REST API

#### Загрузка файла:
```bash
curl -X POST \
  -F "file=@video.mp4" \
  http://localhost:9001/upload
```

#### Получение статистики:
```bash
curl http://localhost:9001/api/stats
```

#### Управление обработкой:
```bash
# Остановка
curl -X POST http://localhost:9001/api/stop

# Возобновление  
curl -X POST http://localhost:9001/api/resume
```

## 🐳 Docker

### Сборка образа:
```bash
docker build -t pig-detect .
```

### Запуск контейнера:
```bash
docker run -p 9001:9001 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/uploads:/app/uploads \
    --gpus all \
    pig-detect
```

### Docker Compose:
```bash
docker-compose up -d
```

## ⚙️ Конфигурация

### Переменные окружения (.env):

```bash
# Flask настройки
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
HOST=0.0.0.0
PORT=9001

# Пути к моделям
MODEL_PATH=models/yolo11n-mod.pt
PT_MODEL_PATH=models/pig_model.pth
TRACKER_CONFIG=models/bytetrack.yaml

# Настройки обработки
DEFAULT_CONF=0.5
MIN_CONFIDENCE=0.6
MIN_AREA=800
MAX_AREA=200000
DEVICE=cuda

# Директории
UPLOAD_DIR=uploads
TEST_DATA_DIR=uploads
MODELS_DIR=models

# Безопасность
MAX_UPLOAD_SIZE=100
RATE_LIMIT_REQUESTS=60
SESSION_TIMEOUT=3600

# Логирование
LOG_LEVEL=INFO
LOG_FILE=app.log
```

### Настройка производительности:

#### Для GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda
```

#### Для CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
export DEVICE=cpu
export TORCH_THREADS=4
```

## 🔧 Troubleshooting

### Проблемы с CUDA:
```bash
# Проверка доступности CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Принудительное использование CPU
export CUDA_VISIBLE_DEVICES=""
```

### Проблемы с камерой:
```bash
# Тест веб-камеры
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Список камер в Linux
ls /dev/video*
```

### Проблемы с моделями:
```bash
# Проверка модели YOLO
python -c "from ultralytics import YOLO; YOLO('models/yolo11n-mod.pt')"

# Проверка PyTorch модели
python -c "import torch; torch.load('models/pig_model.pth')"
```

### Логи и отладка:
```bash
# Просмотр логов
tail -f app.log

# Отладочный режим
export FLASK_DEBUG=True
export LOG_LEVEL=DEBUG
```

### Проблемы с памятью:
```bash
# Ограничение памяти PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Очистка кеша CUDA
python -c "import torch; torch.cuda.empty_cache()"
```

## 📝 Примеры использования

### Запуск с различными источниками:

```bash
# Веб-камера
python run.py --video webcam

# Видеофайл  
python run.py --video uploads/pig_video.mp4

# IP камера
python run.py --video rtsp://192.168.1.100:554/stream

# Изображение
python run.py --video test_image.jpg
```

### Настройка параметров детекции:

```bash
# Высокая точность
python run.py --video webcam --conf 0.8

# Быстрая обработка
python run.py --video webcam --conf 0.3 --device cpu
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создание feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменений (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Создание Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 👥 Авторы

- Основной разработчик - ch1ppyone
- Вклад в ML модели - ch1ppyone
- UI/UX дизайн - ch1ppyone

## 🙏 Благодарности

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) за отличную библиотеку детекции
- [ByteTrack](https://github.com/ifzhang/ByteTrack) за алгоритм трекинга
- [Flask](https://flask.palletsprojects.com/) за веб-фреймворк
- [PyTorch](https://pytorch.org/) за ML фреймворк

---

## 📞 Поддержка

Если у вас возникли вопросы или проблемы:

1. Проверьте [Troubleshooting](#troubleshooting)
2. Создайте Issue в репозитории
3. Обратитесь к документации API

**Система готова к работе! 🚀**