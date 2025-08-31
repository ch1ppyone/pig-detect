"""
Модуль обработки видеопотока для системы мониторинга свиней.
Содержит классы для детекции, классификации и трекинга объектов.
"""

import os
import time
import datetime
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from .detector import Detector
from .db import DatabaseManager
from .logging import logger

# Оптимизация OpenCV
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@dataclass
class ProcessingStats:
    """Статистика обработки."""
    frames_processed: int = 0
    detections_count: int = 0
    classifications_count: int = 0
    errors_count: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def uptime(self) -> float:
        """Время работы в секундах."""
        return time.time() - self.start_time
    
    @property
    def fps(self) -> float:
        """Средний FPS обработки."""
        return self.frames_processed / max(self.uptime, 1)


class VideoProcessor:
    """
    Обработка видеопотока/изображения: YOLO-детекция, PyTorch-классификация, логирование.
    """

    def __init__(
            self,
            model_path: str,
            device: str,
            video_path: str,
            tracker_config: str,
            pt_model_path: str,
            db_path: str,
            conf: float = 0.5
    ):
        """
        Инициализация процессора видео.
        
        Args:
            model_path: Путь к модели YOLO
            device: Устройство для вычислений ('cpu' или 'cuda')
            video_path: Путь к видео/изображению или 'webcam'
            tracker_config: Путь к конфигурации трекера
            pt_model_path: Путь к модели PyTorch для классификации
            db_path: Путь к базе данных
            conf: Порог уверенности детекции
        """
        # Основные параметры
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.video_path = video_path
        self.tracker_config = tracker_config
        self.pt_model_path = pt_model_path
        self.db_path = db_path
        self.conf = float(conf)
        
        # Определяем тип источника
        self.is_image = video_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        self.is_webcam = video_path.lower() in ['webcam', '0', '1', '2', '3']
        
        # Статистика и состояние
        self.stats = ProcessingStats()
        self.running = True
        self._cap_lock = threading.RLock()
        
        # Данные трекинга
        self.logs: List[Tuple[int, str]] = []
        self.track_labels: Dict[int, Dict[str, int]] = {}
        self.state_durations: Dict[int, Dict[str, float]] = {}
        
        # Настройки обработки
        self.conf_threshold = max(0.3, conf)
        self.delay = 1 / 30  # FPS=30
        
        # Логирование параметров
        logger.info(f"🔧 Инициализация VideoProcessor:")
        logger.info(f"   📁 Модель YOLO: {model_path}")
        logger.info(f"   🖥️ Устройство: {self.device}")
        logger.info(f"   📹 Источник: {video_path}")
        logger.info(f"   🎯 Порог уверенности: {self.conf}")
        
        # Инициализация компонентов
        self._init_video_source()
        self._init_detector()
        self._init_database()
        self._init_pytorch_model()
        
        logger.info("✅ VideoProcessor успешно инициализирован")
    
    def _init_video_source(self) -> None:
        """Инициализация источника видео."""
        # Проверка файла (пропускаем для веб-камеры)
        if not self.is_webcam and not os.path.exists(self.video_path):
            raise ValueError(f"Файл не найден: {self.video_path}")
        
        # Инициализация видео/изображения
        if self.is_webcam:
            self._init_webcam()
        elif self.is_image:
            self._init_image()
        else:
            self._init_video_file()
    
    def _init_webcam(self) -> None:
        """Инициализация веб-камеры."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Не удалось открыть веб-камеру")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info(f"📷 Веб-камера открыта, FPS: {fps}")
        self.image = None
    
    def _init_image(self) -> None:
        """Инициализация изображения."""
        self.image = cv2.imread(self.video_path)
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {self.video_path}")
        
        logger.info(f"🖼️ Изображение открыто: {self.video_path}")
        self.cap = None
    
    def _init_video_file(self) -> None:
        """Инициализация видеофайла."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"📹 Видео открыто: {self.video_path}, FPS: {fps}")
        self.image = None
    
    def _init_detector(self) -> None:
        """Инициализация детектора."""
        try:
            self.detector = Detector(self.model_path, str(self.device), self.tracker_config)
            logger.info("✅ Детектор YOLO инициализирован")
        except Exception as e:
            logger.error(f"🚨 Ошибка инициализации детектора: {str(e)}")
            raise
    
    def _init_database(self) -> None:
        """Инициализация базы данных."""
        try:
            # Используем singleton из StateManager вместо создания новой БД
            from .routes import StateManager
            self.db = StateManager.get_db()
            self.state_translations = self.db.get_translations('ru')
            self.class_names = self.db.load_class_names()
            logger.info(f"✅ База данных подключена, переводов: {len(self.state_translations)}")
        except Exception as e:
            logger.error(f"🚨 Ошибка подключения к БД: {str(e)}")
            raise
    
    def _init_pytorch_model(self) -> None:
        """Инициализация модели PyTorch для классификации."""
        try:
            from torchvision import models
            
            # Настройка шрифта для русского текста
            self._init_font()
            
            num_classes = len(self.class_names)
            
            # Инициализация модели
            if not os.path.exists(self.pt_model_path):
                logger.warning(f"⚠️ Модель PyTorch не найдена: {self.pt_model_path}")
                self.model = None
            else:
                self._load_pytorch_model(num_classes)
            
            # Преобразования для предобработки
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        except Exception as e:
            logger.error(f"🚨 Ошибка инициализации PyTorch модели: {str(e)}")
            self.model = None
            self.transform = None
    
    def _init_font(self) -> None:
        """Инициализация шрифта для отображения русского текста."""
        font_paths = [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf"
        ]
        
        for font_path in font_paths:
            try:
                self.font = ImageFont.truetype(font_path, 20)
                logger.debug(f"✅ Загружен шрифт: {font_path}")
                return
            except Exception:
                continue
        
        # Fallback на дефолтный шрифт
        self.font = ImageFont.load_default()
        logger.warning("⚠️ Используется дефолтный шрифт")
    
    def _load_pytorch_model(self, num_classes: int) -> None:
        """Загрузка модели PyTorch."""
        from torchvision import models
        
        try:
            # Создаем модель с правильной архитектурой
            self.model = models.mobilenet_v3_small(weights=None)
            
            # Заменяем классификатор на новую архитектуру (как в train_classify.py)
            self.model.classifier[3] = torch.nn.Sequential(
                torch.nn.Linear(self.model.classifier[3].in_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.4),
                torch.nn.Linear(256, num_classes)
            )
            
            # Пытаемся загрузить веса
            try:
                state_dict = torch.load(self.pt_model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"✅ PyTorch модель загружена с {num_classes} классами")
            except Exception as load_error:
                logger.error(f"🚨 Ошибка загрузки весов: {str(load_error)}")
                logger.warning("⚠️ Используем модель с случайными весами")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"🚨 Ошибка создания модели: {str(e)}")
            self.model = None

    def reset_video(self):
        """Перезапуск видеопотока."""
        if self.is_image:
            logger.info("🔄 Изображение не требует перезапуска")
            return True
        with self._cap_lock:
            self.cap.release()
            if self.is_webcam:
                self.cap = cv2.VideoCapture(0)  # Веб-камера
                if not self.cap.isOpened():
                    logger.error("🚨 Не удалось перезапустить веб-камеру")
                    return False
                logger.info("✅ Веб-камера перезапущена")
            else:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    logger.error(f"🚨 Не удалось перезапустить видео: {self.video_path}")
                    return False
                logger.info(f"✅ Видео перезапущено: {self.video_path}")
            return True

    def filter_pig_detections(self, boxes, confidences):
        """
        Минимальная фильтрация - пропускаем почти все детекции.
        
        Args:
            boxes: Список боксов детекции [x1, y1, x2, y2]
            confidences: Список уверенности детекции
            
        Returns:
            Список индексов валидных детекций
        """
        valid_indices = []
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            # Только базовая проверка уверенности
            if conf < self.conf_threshold:
                logger.debug(f"🚫 Бокс {i} отклонен: низкая уверенность {conf:.2f} < {self.conf_threshold}")
                continue
            
            # Проверка на очень маленькие боксы (артефакты)
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if area < 100:  # Только самые мелкие артефакты
                logger.debug(f"🚫 Бокс {i} отклонен: слишком маленький {area}")
                continue
            
            valid_indices.append(i)
            logger.debug(f"✅ Бокс {i} принят: area={area}, conf={conf:.2f}")
        
        logger.info(f"🎯 Минимальная фильтрация: {len(valid_indices)}/{len(boxes)} боксов прошли проверку")
        return valid_indices

    def draw_russian_text(self, img, text, position, color=(0, 255, 0)):
        """
        Отрисовка русского текста на изображении с помощью PIL.
        
        Args:
            img: Изображение OpenCV (BGR)
            text: Текст для отрисовки
            position: Позиция (x, y)
            color: Цвет в формате RGB
        
        Returns:
            Изображение с нанесенным текстом
        """
        try:
            # Конвертируем BGR в RGB для PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            # Рисуем текст
            x, y = position
            draw.text((x, y), text, font=self.font, fill=color)
            
            # Конвертируем обратно в BGR для OpenCV
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            logger.warning(f"⚠️ Ошибка отрисовки русского текста: {e}")
            # Fallback на стандартный OpenCV
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img

    def preprocess_crop(self, roi):
        """Предобработка области интереса для PyTorch."""
        if roi.size == 0:
            return torch.zeros((3, 224, 224), device=self.device)
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self.transform(img).to(self.device)
        return img

    def classify_pig_states(self, frame: np.ndarray, bboxes):
        """Классификация состояний свиней с PyTorch. Возвращает список той же длины, что и bboxes."""
        try:
            # Если модель не загружена, возвращаем случайные состояния для демонстрации
            if self.model is None:
                logger.warning("⚠️ PyTorch модель не загружена, возвращаем случайные состояния")
                import random
                return [random.randint(0, len(self.class_names)-1) if len(self.class_names) > 0 else 0 for _ in bboxes]
            
            crops = []
            orig_indices = []
            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    logger.warning(f"⚠️ Слишком маленький бокс: {bbox.tolist()}")
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.warning(f"⚠️ Пустой crop для бокса: {bbox.tolist()}")
                    continue
                crops.append(self.preprocess_crop(crop))
                orig_indices.append(idx)

            states_aligned = [None] * len(bboxes)
            if not crops:
                logger.warning("⚠️ Нет валидных crops для классификации")
                return states_aligned

            # Batch-processing
            crops = torch.stack(crops)
            with torch.no_grad():
                preds = self.model(crops)
            state_indices = torch.argmax(preds, dim=1).cpu().numpy().tolist()
            for i, idx in enumerate(orig_indices):
                if i < len(state_indices):
                    states_aligned[idx] = state_indices[i]
            return states_aligned

        except Exception as e:
            logger.error(f"🚨 Ошибка классификации: {str(e)}")
            return [None] * len(bboxes)

    def process_frame(self):
        """Обработка одного кадра с детекцией и классификацией."""
        import time
        start_time = time.time()

        if self.is_image:
            frame = self.image
            if frame is None:
                logger.error("🚨 Изображение не загружено")
                return None
        else:
            with self._cap_lock:
                ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("⚠️ Не удалось прочитать кадр")
                if not self.reset_video():
                    return None
                with self._cap_lock:
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.error("🚨 Не удалось прочитать кадр после перезапуска")
                    return None

        try:
            results = self.detector.detect_and_track(frame, conf=self.conf)
            if not results or not len(results) or results[0] is None:
                logger.warning("⚠️ Нет результатов детекции")
                return frame

            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
                logger.warning("⚠️ Нет боксов в результатах детекции")
                return frame

            # Проверяем, что это детекционная модель (не сегментационная)
            if hasattr(result, 'masks') and result.masks is not None:
                logger.warning("⚠️ Обнаружена сегментационная модель, но ожидается детекционная")
                frame_drawn = result.plot(boxes=True, masks=True, labels=False)
            else:
                # Стандартное отображение для детекционной модели
                frame_drawn = result.plot(boxes=True, labels=False)
            # Извлекаем все боксы и уверенности для фильтрации
            boxes_iter = list(result.boxes)
            all_bboxes = [box.xyxy[0].cpu().numpy() for box in boxes_iter]
            all_confidences = [box.conf.item() for box in boxes_iter]
            all_track_ids = [int(box.id.item()) if getattr(box, 'id', None) is not None else idx for idx, box in enumerate(boxes_iter)]

            logger.info(f"🔍 Найдено {len(all_bboxes)} объектов до фильтрации")

            # Применяем фильтрацию для повышения точности
            valid_indices = self.filter_pig_detections(all_bboxes, all_confidences)
            
            # Оставляем только валидные детекции
            bboxes = [all_bboxes[i] for i in valid_indices]
            track_ids = [all_track_ids[i] for i in valid_indices]

            logger.info(f"🎯 Осталось {len(bboxes)} объектов после фильтрации для классификации")

            if bboxes and len(self.class_names) > 0:
                state_indices = self.classify_pig_states(frame, bboxes)
                successful_classifications = 0
                
                for track_id, state_idx, bbox in zip(track_ids, state_indices, bboxes):
                    if state_idx is None:
                        logger.warning(f"⚠️ Пропущена классификация для track_id {track_id}")
                        continue
                    
                    if state_idx >= len(self.class_names):
                        logger.warning(f"⚠️ Неверный индекс состояния {state_idx} для track_id {track_id}")
                        continue
                        
                    state = self.class_names[state_idx]
                    state_desc = self.db.get_states().get(state, state)  # Используем сам код если описания нет
                    state_ru = self.state_translations.get(state, state)  # Получаем русский перевод
                    
                    self.track_labels.setdefault(track_id, {}).update(
                        {state: self.track_labels.get(track_id, {}).get(state, 0) + 1}
                    )
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or 30 if not self.is_image else 30
                    self.state_durations.setdefault(track_id, {}).update(
                        {state: self.state_durations.get(track_id, {}).get(state, 0) + 1 / fps}
                    )
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.logs.append([track_id, f"{timestamp} - Свинья {track_id} в состоянии {state_ru}"])
                    logger.info(f"🏷️ Объект: track_id={track_id}, state={state_ru} ({state}), bbox={bbox.tolist()}")
                    
                    # Рисуем подписи состояния на кадре (на русском языке)
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Используем PIL для корректного отображения русского текста
                    try:
                        frame_drawn = self.draw_russian_text(
                            frame_drawn, 
                            f"{track_id}:{state_ru}",
                            (x1, max(0, y1 - 25)),
                            color=(0, 255, 0)  # RGB формат для PIL
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка отрисовки русского текста: {e}, используем ASCII")
                        # Fallback на ASCII если не получается отрисовать русский текст
                        cv2.putText(
                            frame_drawn,
                            f"{track_id}:{state}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                    successful_classifications += 1
                
                logger.info(f"✅ Успешно классифицировано {successful_classifications} из {len(bboxes)} объектов")
            else:
                logger.warning("⚠️ Нет боксов или классов для классификации")

            logger.info(f"🕒 Обработка кадра завершена за {time.time() - start_time:.2f} сек")
            return frame_drawn

        except Exception as e:
            logger.error(f"🚨 Ошибка обработки кадра: {str(e)}")
            import traceback
            logger.error(f"🚨 Полная трассировка: {traceback.format_exc()}")
            return frame

    def next_frame(self):
        """Получение следующего кадра для SocketIO."""
        return self.process_frame()

    def release(self):
        """Освобождение ресурсов."""
        if not self.is_image:
            with self._cap_lock:
                self.cap.release()
        self.running = False
        # БД больше не закрываем, так как это singleton
        logger.info("🛑 VideoProcessor остановлен")

    def __del__(self):
        """Очистка при удалении объекта."""
        if not self.is_image and hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        # БД больше не закрываем, так как это singleton
        logger.info("🧹 Ресурсы VideoProcessor освобождены")