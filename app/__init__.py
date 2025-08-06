import torch
from flask import Flask
from flask_socketio import SocketIO
from queue import Queue
from .config import check_hardware, create_video_processor
from .logging import logger

app = Flask(__name__, template_folder="../template", static_folder="../static")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*")

# Глобальные переменные
frame_queue = Queue(maxsize=50)  # Увеличен для стабильности
processors = {}
background_tasks = {}


def create_app(processor=None):
    """
    Фабричная функция для создания приложения Flask.

    Args:
        processor: Экземпляр VideoProcessor (опционально).

    Returns:
        Flask: Настроенное приложение Flask.
    """
    # Проверка оборудования
    device = check_hardware()

    if processor is None:
        try:
            processor = create_video_processor()
            processors["default"] = processor
            logger.info("✅ Процессор по умолчанию успешно создан")
        except Exception as e:
            logger.error(f"🚨 Ошибка при создании процессора по умолчанию: {str(e)}")
            raise

    from . import routes
    return app


from . import routes