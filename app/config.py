import os
import argparse
import torch
from dotenv import load_dotenv
from .video_processor import VideoProcessor
from .logging import logger

# Загрузка конфигурации из .env
load_dotenv()

# Парсинг аргументов
parser = argparse.ArgumentParser(description="Система мониторинга поведения свиней")
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help="Устройство для обработки (cpu или cuda)")
parser.add_argument('--imgsz', type=int, default=int(os.getenv('DEFAULT_IMGSZ', 640)), help="Размер изображения для YOLO")
parser.add_argument('--frame-skip', type=int, default=1, help="Количество пропускаемых кадров")
args = parser.parse_args()

# Конфигурационные параметры из .env
MODEL_PATH = os.getenv('MODEL_PATH', 'models/yolo11m-seg-mod.pt')
DB_PATH = os.getenv('DB_PATH', 'pig_states.db')
PT_MODEL_PATH = os.getenv('PT_MODEL_PATH', 'models/pig_model.pth')
VIDEO_PATH = os.getenv('VIDEO_PATH', 'test_data/pigs.mp4')
TEST_VIDEO_PATH = os.getenv('TEST_VIDEO_PATH', 'test_data/2.mp4')
DEFAULT_CONF = float(os.getenv('DEFAULT_CONF', 0.5))
TRACKER_CONFIG = os.getenv('TRACKER_CONFIG', 'models/bytetrack.yaml')
MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 100)) * 1024 * 1024

logger.info(f"📂 Путь к модели YOLO: {MODEL_PATH}")
logger.info(f"📂 Путь к базе данных: {DB_PATH}")
logger.info(f"📂 Путь к модели PyTorch: {PT_MODEL_PATH}")
logger.info(f"📂 Путь к видео: {VIDEO_PATH}")
logger.info(f"📂 Путь к тестовому видео: {TEST_VIDEO_PATH}")
logger.info(f"🔧 Конфигурация трекера: {TRACKER_CONFIG}")

# Выбор устройства
device = 'cpu' if args.device == 'cpu' or not torch.cuda.is_available() else 'cuda'
if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logger.info("✅ Запуск на CPU активирован")
else:
    logger.info("✅ Запуск на GPU активирован")

def check_hardware():
    """Проверка оборудования и версий библиотек."""
    logger.info(f"🔥 Версия PyTorch: {torch.__version__}")
    logger.info(f"⚡ CUDA доступна (PyTorch): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 Устройство CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"🔧 Используемое устройство: {device}")
    return device

def create_video_processor(video_path=VIDEO_PATH):
    """Фабричная функция для создания VideoProcessor."""
    try:
        processor = VideoProcessor(
            model_path=MODEL_PATH,
            device=device,
            video_path=video_path,
            tracker_config=TRACKER_CONFIG,
            pt_model_path=PT_MODEL_PATH,
            db_path=DB_PATH
        )
        return processor
    except Exception as e:
        logger.error(f"🚨 Ошибка создания VideoProcessor: {str(e)}")
        raise

def get_color_for_track(track_id):
    """Генерация цвета для трека."""
    import random
    random.seed(track_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )

def is_safe_filename(filename):
    """Проверка безопасности имени файла."""
    import re
    return bool(re.match(r'^[\w\-\.][\w\-\.]*[\w\-\.]$', filename)) and '..' not in filename