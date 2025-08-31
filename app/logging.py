"""
Модуль конфигурации логирования для системы мониторинга свиней.
Содержит настройку логгеров с ротацией файлов и форматированием.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли."""
    
    # Цветовые коды ANSI
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Форматирование записи с цветами."""
        if sys.stdout.isatty():  # Только для терминала
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class LoggerManager:
    """Менеджер системы логирования."""
    
    def __init__(self, 
                 name: str = 'pig_monitor',
                 level: str = 'INFO',
                 log_file: Optional[str] = None,
                 max_size_mb: int = 10,
                 backup_count: int = 5):
        """
        Инициализация менеджера логирования.
        
        Args:
            name: Имя логгера
            level: Уровень логирования
            log_file: Путь к файлу логов (опционально)
            max_size_mb: Максимальный размер файла лога в MB
            backup_count: Количество backup файлов
        """
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.log_file = log_file
        self.max_size_mb = max_size_mb
        self.backup_count = backup_count
        
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Очищаем существующие обработчики
        logger.handlers.clear()
        
        # Форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Консольный обработчик с цветами
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console_handler)
        
        # Файловый обработчик с ротацией (если указан файл)
        if self.log_file:
            try:
                # Создаем директорию для логов
                log_dir = os.path.dirname(self.log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_size_mb * 1024 * 1024,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(self.level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                logger.error(f"🚨 Не удалось настроить файловое логирование: {str(e)}")
        
        return logger
    
    @property
    def logger(self) -> logging.Logger:
        """Получение настроенного логгера."""
        return self._logger


# Создаем глобальный логгер
_logger_manager = LoggerManager()
logger = _logger_manager.logger

# Функция для обновления конфигурации логирования
def configure_logging(level: str = 'INFO', 
                     log_file: Optional[str] = None,
                     max_size_mb: int = 10,
                     backup_count: int = 5) -> None:
    """
    Перенастройка системы логирования.
    
    Args:
        level: Уровень логирования
        log_file: Путь к файлу логов
        max_size_mb: Максимальный размер файла
        backup_count: Количество backup файлов
    """
    global _logger_manager, logger
    
    _logger_manager = LoggerManager(
        level=level,
        log_file=log_file,
        max_size_mb=max_size_mb,
        backup_count=backup_count
    )
    logger = _logger_manager.logger