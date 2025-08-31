"""
Конфигурационный модуль для системы мониторинга свиней.
Содержит все настройки, валидацию и фабричные методы.
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

from .logging import logger

# Загрузка переменных окружения
load_dotenv()


@dataclass
class AppConfig:
    """Конфигурация приложения."""
    # Flask настройки
    secret_key: str
    flask_env: str
    flask_debug: bool
    host: str
    port: int
    
    # Пути к файлам и моделям
    model_path: str
    db_path: str
    pt_model_path: str
    video_path: str
    test_video_path: str
    tracker_config: str
    
    # Директории
    upload_dir: str
    test_data_dir: str
    models_dir: str
    
    # Настройки обработки
    default_conf: float
    min_confidence: float
    min_area: int
    max_area: int
    device: str
    torch_threads: int
    
    # Ограничения и безопасность
    max_upload_size: int
    db_timeout: int
    max_workers: int
    rate_limit_requests: int
    rate_limit_window: int
    session_timeout: int
    
    # Расширения файлов
    allowed_video_extensions: List[str]
    allowed_image_extensions: List[str]
    
    # Логирование
    log_level: str
    log_file: str
    log_max_size_mb: int
    log_backup_count: int


class ConfigManager:
    """Менеджер конфигурации."""
    
    def __init__(self):
        self._config = self._load_config()
        self._device = self._setup_device()
        
    def _load_config(self) -> AppConfig:
        """Загрузка конфигурации из переменных окружения."""
        return AppConfig(
            # Flask настройки
            secret_key=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
            flask_env=os.getenv('FLASK_ENV', 'development'),
            flask_debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', '9001')),
            
            # Пути к файлам и моделям
            model_path=os.getenv('MODEL_PATH', 'models/yolo11n-mod.pt'),
            db_path=os.getenv('DB_PATH', 'pig_states.db'),
            pt_model_path=os.getenv('PT_MODEL_PATH', 'models/pig_model.pth'),
            video_path=os.getenv('VIDEO_PATH', 'webcam'),
            test_video_path=os.getenv('TEST_VIDEO_PATH', 'webcam'),
            tracker_config=os.getenv('TRACKER_CONFIG', 'models/bytetrack.yaml'),
            
            # Директории
            upload_dir=os.getenv('UPLOAD_DIR', 'uploads'),
            test_data_dir=os.getenv('TEST_DATA_DIR', 'uploads'),
            models_dir=os.getenv('MODELS_DIR', 'models'),
            
            # Настройки обработки
            default_conf=float(os.getenv('DEFAULT_CONF', '0.5')),
            min_confidence=float(os.getenv('MIN_CONFIDENCE', '0.6')),
            min_area=int(os.getenv('MIN_AREA', '800')),
            max_area=int(os.getenv('MAX_AREA', '200000')),
            device=os.getenv('DEVICE', 'cuda'),
            torch_threads=int(os.getenv('TORCH_THREADS', '4')),
            
            # Ограничения и безопасность
            max_upload_size=int(os.getenv('MAX_UPLOAD_SIZE', '100')) * 1024 * 1024,
            db_timeout=int(os.getenv('DB_TIMEOUT', '30')),
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '60')),
            rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
            session_timeout=int(os.getenv('SESSION_TIMEOUT', '3600')),
            
            # Расширения файлов
            allowed_video_extensions=os.getenv('ALLOWED_VIDEO_EXTENSIONS', 'mp4,avi,mov,mkv').split(','),
            allowed_image_extensions=os.getenv('ALLOWED_IMAGE_EXTENSIONS', 'jpg,jpeg,png,bmp').split(','),
            
            # Логирование
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'app.log'),
            log_max_size_mb=int(os.getenv('LOG_MAX_SIZE_MB', '10')),
            log_backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5')),
        )
    
    def _setup_device(self) -> str:
        """Настройка вычислительного устройства."""
        if self._config.device == 'cpu' or not torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = 'cpu'
            logger.info("✅ Запуск на CPU активирован")
        else:
            device = 'cuda'
            logger.info("✅ Запуск на GPU активирован")
        
        torch.set_num_threads(self._config.torch_threads)
        return device

    
    @property
    def config(self) -> AppConfig:
        """Получение конфигурации."""
        return self._config
    
    @property
    def device(self) -> str:
        """Получение устройства."""
        return self._device
    
    def log_configuration(self) -> None:
        """Логирование текущей конфигурации."""
        logger.info(f"📂 Путь к модели YOLO: {self._config.model_path}")
        logger.info(f"📂 Путь к базе данных: {self._config.db_path}")
        logger.info(f"📂 Путь к модели PyTorch: {self._config.pt_model_path}")
        logger.info(f"📂 Путь к видео: {self._config.video_path}")
        logger.info(f"📂 Путь к тестовому видео: {self._config.test_video_path}")
        logger.info(f"🔧 Конфигурация трекера: {self._config.tracker_config}")
    
    def check_hardware(self) -> str:
        """Проверка оборудования и версий библиотек."""
        logger.info(f"🔥 Версия PyTorch: {torch.__version__}")
        logger.info(f"⚡ CUDA доступна (PyTorch): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 Устройство CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"🔧 Используемое устройство: {self._device}")
        return self._device
    
    def validate_config(self) -> bool:
        """Валидация конфигурации при запуске."""
        errors = []
        
        # Проверка SECRET_KEY
        if self._config.secret_key == 'dev-secret-key-change-in-production':
            if self._config.flask_env == 'production':
                errors.append("SECRET_KEY должен быть изменен в продакшн окружении")
            else:
                logger.warning("⚠️ Используется дефолтный SECRET_KEY в dev окружении")
        
        # Проверка путей к моделям
        if not os.path.exists(self._config.model_path):
            logger.warning(f"⚠️ YOLO модель не найдена: {self._config.model_path}")
        
        if not os.path.exists(self._config.pt_model_path):
            logger.warning(f"⚠️ PyTorch модель не найдена: {self._config.pt_model_path}")
        
        # Проверка и создание директорий
        for dir_path in [self._config.upload_dir, self._config.test_data_dir, self._config.models_dir]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"✅ Создана директория: {dir_path}")
                except Exception as e:
                    errors.append(f"Не удалось создать директорию {dir_path}: {str(e)}")
        
        # Проверка настроек
        if not (0.0 <= self._config.default_conf <= 1.0):
            errors.append("DEFAULT_CONF должен быть в диапазоне [0.0, 1.0]")
        
        if not (0.0 <= self._config.min_confidence <= 1.0):
            errors.append("MIN_CONFIDENCE должен быть в диапазоне [0.0, 1.0]")
        
        if self._config.min_area >= self._config.max_area:
            errors.append("MIN_AREA должен быть меньше MAX_AREA")
        
        if errors:
            logger.error("🚨 Ошибки конфигурации:")
            for error in errors:
                logger.error(f"   - {error}")
            return False
        
        logger.info("✅ Конфигурация валидна")
        return True
    
    def create_video_processor(self, video_path: Optional[str] = None, conf: Optional[float] = None):
        """Фабричная функция для создания VideoProcessor."""
        from .video_processor import VideoProcessor
        
        try:
            if video_path is None:
                video_path = self._config.video_path
            if conf is None:
                conf = self._config.default_conf
                
            processor = VideoProcessor(
                model_path=self._config.model_path,
                device=self._device,
                video_path=video_path,
                tracker_config=self._config.tracker_config,
                pt_model_path=self._config.pt_model_path,
                db_path=self._config.db_path,
                conf=conf
            )
            return processor
        except Exception as e:
            logger.error(f"🚨 Ошибка создания VideoProcessor: {str(e)}")
            raise


class SecurityUtils:
    """Утилиты безопасности."""
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Проверка безопасности имени файла."""
        if not filename or '..' in filename:
            return False
        # Разрешаем специальные значения для источников видео
        special_sources = {'webcam', 'remote_camera'}
        if filename in special_sources:
            return True
        # Для обычных файлов используем более мягкую валидацию
        return bool(re.match(r'^[\w\-\.]+$', filename))
    
    @staticmethod
    def get_color_for_track(track_id: int) -> Tuple[int, int, int]:
        """Генерация цвета для трека."""
        random.seed(track_id)
        return (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )


# Глобальный экземпляр менеджера конфигурации
config_manager = ConfigManager()

# Обратная совместимость - экспорт старых переменных
MODEL_PATH = config_manager.config.model_path
DB_PATH = config_manager.config.db_path
PT_MODEL_PATH = config_manager.config.pt_model_path
VIDEO_PATH = config_manager.config.video_path
TEST_VIDEO_PATH = config_manager.config.test_video_path
DEFAULT_CONF = config_manager.config.default_conf
TRACKER_CONFIG = config_manager.config.tracker_config
device = config_manager.device

# Утилиты для обратной совместимости
def check_hardware():
    return config_manager.check_hardware()

def create_video_processor(video_path=None, conf=None):
    return config_manager.create_video_processor(video_path, conf)

def is_safe_filename(filename):
    return SecurityUtils.is_safe_filename(filename)

def get_color_for_track(track_id):
    return SecurityUtils.get_color_for_track(track_id)

# Логирование конфигурации при импорте
config_manager.log_configuration()