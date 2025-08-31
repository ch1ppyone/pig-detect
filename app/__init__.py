"""
Главный модуль приложения Flask для системы мониторинга свиней.
Содержит фабричную функцию создания приложения и глобальные переменные.
"""

from typing import Optional, Dict, Any
from queue import Queue

import torch
from flask import Flask
from flask_socketio import SocketIO

from .config import config_manager
from .logging import logger, configure_logging


class AppFactory:
    """Фабрика для создания Flask приложения."""
    
    @staticmethod
    def create_app(processor=None) -> Flask:
        """
        Создание и настройка Flask приложения.

        Args:
            processor: Экземпляр VideoProcessor (опционально)

        Returns:
            Настроенное приложение Flask
        """
        # Настройка логирования из конфигурации
        configure_logging(
            level=config_manager.config.log_level,
            log_file=config_manager.config.log_file,
            max_size_mb=config_manager.config.log_max_size_mb,
            backup_count=config_manager.config.log_backup_count
        )
        
        # Создание приложения
        app = Flask(__name__, template_folder="../template", static_folder="../static")
        
        # Конфигурация Flask
        app.config.update({
            'MAX_CONTENT_LENGTH': config_manager.config.max_upload_size,
            'SECRET_KEY': config_manager.config.secret_key,
            'SESSION_TYPE': 'filesystem',
            'PERMANENT_SESSION_LIFETIME': config_manager.config.session_timeout
        })
        
        # Используем глобальный экземпляр SocketIO
        socketio.init_app(app, cors_allowed_origins="*")
        
        # Проверка оборудования
        device = config_manager.check_hardware()
        
        # Валидация конфигурации
        if not config_manager.validate_config():
            logger.error("🚨 Конфигурация содержит ошибки")
            raise RuntimeError("Некорректная конфигурация")

        # Создание процессора по умолчанию
        if processor is None:
            try:
                processor = config_manager.create_video_processor()
                processors["default"] = processor
                logger.info("✅ Процессор по умолчанию создан")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать процессор: {str(e)}")
                logger.info("ℹ️ Приложение продолжит работу без видеопроцессора")

        # Регистрация маршрутов и аутентификации
        AppFactory._register_components(app)
        
        return app
    
    @staticmethod
    def _register_components(app: Flask) -> None:
        """Регистрация компонентов приложения."""
        # Инициализация аутентификации
        from .auth import init_auth
        init_auth(app, config_manager.config.db_path)
        
        # Импортируем и регистрируем маршруты  
        from .routes import register_routes, register_socketio_events
        register_routes(app)
        register_socketio_events(socketio)  # Передаем тот же экземпляр
        
        logger.info("✅ Компоненты приложения зарегистрированы")


# Глобальные переменные для совместимости
app = Flask(__name__, template_folder="../template", static_folder="../static")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SECRET_KEY'] = config_manager.config.secret_key
socketio = SocketIO(app, cors_allowed_origins="*")

frame_queue: Queue = Queue(maxsize=50)
processors: Dict[str, Any] = {}
background_tasks: Dict[str, Any] = {}

# Функция для обратной совместимости
def create_app(processor=None):
    """Создание приложения (обратная совместимость)."""
    return AppFactory.create_app(processor)