"""
Модуль детекции объектов с использованием YOLO.
Содержит класс Detector для обнаружения и трекинга свиней.
"""

from typing import Dict, List, Optional, Any
import os

import numpy as np
from ultralytics import YOLO
from .logging import logger


class DetectorError(Exception):
    """Исключение для ошибок детектора."""
    pass


class Detector:
    """Детектор объектов на базе YOLO."""
    
    def __init__(self, model_path: str, device: str, tracker_config: str = "bytetrack.yaml"):
        """
        Инициализация детектора YOLO.

        Args:
            model_path: Путь к модели YOLO
            device: Устройство ('cuda' или 'cpu')
            tracker_config: Конфигурация трекера
            
        Raises:
            DetectorError: При ошибке инициализации модели
        """
        self.model_path = model_path
        self.device = device
        self.tracker_config = tracker_config
        self.tracker_available = False  # Отключено из-за проблем с fuse_score
        
        logger.info(f"🔧 Инициализация детектора YOLO:")
        logger.info(f"   📁 Модель: {model_path}")
        logger.info(f"   🖥️ Устройство: {device}")
        logger.info(f"   🎯 Трекер: {tracker_config}")
        
        self._init_model()
    
    def _init_model(self) -> None:
        """Инициализация модели YOLO."""
        if not os.path.exists(self.model_path):
            raise DetectorError(f"Модель не найдена: {self.model_path}")
        
        try:
            self.detector = YOLO(self.model_path)
            self.detector.to(self.device)
            
            # Проверяем тип модели
            model_type = getattr(self.detector.model, 'task', 'unknown')
            if 'segment' in str(model_type).lower():
                logger.warning("⚠️ Загружена сегментационная модель, но ожидается детекционная")
            else:
                logger.info(f"✅ Детекционная YOLO-модель загружена на {self.device}")
                
        except Exception as e:
            logger.error(f"🚨 Ошибка инициализации YOLO: {str(e)}")
            raise DetectorError(f"Ошибка инициализации модели: {str(e)}") from e

    def detect_and_track(self, frame: np.ndarray, conf: float = 0.5) -> Optional[List[Any]]:
        """
        Детекция и трекинг объектов на кадре.

        Args:
            frame: Входной кадр (numpy array)
            conf: Порог уверенности для детекции

        Returns:
            Список результатов детекции или None при ошибке
            
        Raises:
            DetectorError: При ошибке детекции
        """
        try:
            # Валидация входных данных
            if frame is None or frame.size == 0:
                logger.warning("⚠️ Получен пустой кадр")
                return None
            
            # Детекция объектов
            results = self.detector.predict(
                frame, 
                conf=conf, 
                verbose=False,
                device=self.device
            )
            
            if not results or not results[0].boxes:
                logger.debug("🔍 Объекты не обнаружены")
                return results
            
            num_objects = len(results[0].boxes)
            logger.debug(f"🔍 Детекция: найдено {num_objects} объектов (conf >= {conf})")
            
            return results
            
        except Exception as e:
            logger.error(f"🚨 Ошибка детекции: {str(e)}")
            raise DetectorError(f"Ошибка детекции: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        try:
            return {
                'model_path': self.model_path,
                'device': self.device,
                'tracker_config': self.tracker_config,
                'tracker_available': self.tracker_available,
                'model_type': getattr(self.detector.model, 'task', 'unknown') if hasattr(self, 'detector') else 'unknown'
            }
        except Exception as e:
            logger.error(f"🚨 Ошибка получения информации о модели: {str(e)}")
            return {}