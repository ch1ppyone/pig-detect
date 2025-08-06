import cv2
import os
import numpy as np
import threading
import torch
import datetime
from torchvision import transforms
from .detector import Detector
from .db import DatabaseManager
from .logging import logger

# Настройки OpenCV
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


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
            db_path: str
    ):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.video_path = video_path
        self.tracker_config = tracker_config
        self.pt_model_path = pt_model_path
        self.db_path = db_path
        self.is_image = video_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

        logger.info(f"🔧 Параметры VideoProcessor: model={model_path}, device={self.device}, "
                    f"video={video_path}, tracker={tracker_config}, pt_model={pt_model_path}, db={db_path}")

        # Проверка файла
        if not os.path.exists(video_path):
            logger.error(f"🚨 Файл не найден: {video_path}")
            raise ValueError(f"Файл не найден: {video_path}")

        # Инициализация детектора
        try:
            self.detector = Detector(model_path, device, tracker_config)
            logger.info("✅ Детектор успешно инициализирован")
        except Exception as e:
            logger.error(f"🚨 Ошибка инициализации детектора: {str(e)}")
            raise

        # Загрузка PyTorch модели
        try:
            from torchvision import models
            self.db = DatabaseManager(db_path=db_path, yolo_model=self.detector)
            num_classes = len(self.db.load_class_names())
            self.model = models.mobilenet_v3_small(num_classes=num_classes)
            self.model.load_state_dict(torch.load(pt_model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ PyTorch модель загружена: {pt_model_path}")
        except Exception as e:
            logger.error(f"🚨 Ошибка загрузки PyTorch модели: {str(e)}")
            raise ValueError(f"Ошибка загрузки PyTorch модели: {str(e)}")

        # Преобразования для предобработки
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Аналог TF [-1, 1]
        ])

        # Загрузка имён классов из БД
        self.class_names = self.db.load_class_names()

        # Инициализация видео/изображения
        if self.is_image:
            self.image = cv2.imread(video_path)
            if self.image is None:
                logger.error(f"🚨 Не удалось загрузить изображение: {video_path}")
                raise ValueError(f"Не удалось загрузить изображение: {video_path}")
            logger.info(f"🖼️ Изображение открыто: {video_path}")
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                logger.error(f"🚨 Не удалось открыть видео: {video_path}")
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"📹 Видео открыто: {video_path}, FPS: {fps}")
            self.image = None

        self._cap_lock = threading.Lock()
        self.logs = []
        self.track_labels = {}
        self.state_durations = {}
        self.running = True
        self.delay = 1 / 30  # FPS=30

    def reset_video(self):
        """Перезапуск видеопотока."""
        if self.is_image:
            logger.info("🔄 Изображение не требует перезапуска")
            return True
        with self._cap_lock:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"🚨 Не удалось перезапустить видео: {self.video_path}")
                return False
            logger.info(f"✅ Видео перезапущено: {self.video_path}")
            return True

    def preprocess_crop(self, roi):
        """Предобработка области интереса для PyTorch."""
        if roi.size == 0:
            return torch.zeros((3, 224, 224), device=self.device)
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self.transform(img).to(self.device)
        return img

    def classify_pig_states(self, frame: np.ndarray, bboxes):
        """Классификация состояний свиней с PyTorch."""
        try:
            crops = []
            valid_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    logger.warning(f"⚠️ Слишком маленький бокс: {bbox.tolist()}")
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.warning(f"⚠️ Пустой crop для бокса: {bbox.tolist()}")
                    continue
                crops.append(self.preprocess_crop(crop))
                valid_bboxes.append(bbox)

            if not crops:
                logger.warning("⚠️ Нет валидных crops для классификации")
                return [None] * len(bboxes)

            # Batch-processing
            crops = torch.stack(crops)
            with torch.no_grad():
                preds = self.model(crops)
            state_indices = torch.argmax(preds, dim=1).cpu().numpy()
            return [state_indices[i] if i < len(state_indices) else None for i in range(len(bboxes))]

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
            results = self.detector.detect_and_track(frame)
            if not results or not len(results) or results[0] is None:
                logger.warning("⚠️ Нет результатов детекции")
                return frame

            frame_drawn = results[0].plot(boxes=True, masks=True, labels=False)
            bboxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if box.id is not None]
            track_ids = [int(box.id) for box in results[0].boxes if box.id is not None]

            if bboxes:
                state_indices = self.classify_pig_states(frame, bboxes)
                for track_id, state_idx, bbox in zip(track_ids, state_indices, bboxes):
                    if state_idx is None or not self.class_names:
                        logger.warning(f"⚠️ Пропущена классификация для track_id {track_id}")
                        continue
                    state = self.class_names[state_idx]
                    state_desc = self.db.get_states().get(state, "Unknown")
                    if state_desc == "Unknown":
                        logger.warning(f"⚠️ Состояние {state} не найдено в базе")
                        continue
                    self.track_labels.setdefault(track_id, {}).update(
                        {state: self.track_labels.get(track_id, {}).get(state, 0) + 1}
                    )
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or 30 if not self.is_image else 30
                    self.state_durations.setdefault(track_id, {}).update(
                        {state: self.state_durations.get(track_id, {}).get(state, 0) + 1 / fps}
                    )
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.logs.append([track_id, f"{timestamp} - Свинья {track_id} в состоянии {state_desc}"])
                    logger.info(f"🏷️ Объект: track_id={track_id}, state={state_desc}, bbox={bbox.tolist()}")

            logger.info(f"🕒 Обработка кадра завершена за {time.time() - start_time:.2f} сек")
            return frame_drawn

        except Exception as e:
            logger.error(f"🚨 Ошибка обработки кадра: {str(e)}")
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
        self.db.close()
        logger.info("🛑 VideoProcessor остановлен")

    def __del__(self):
        """Очистка при удалении объекта."""
        if not self.is_image and hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'db'):
            self.db.close()
        logger.info("🧹 Ресурсы VideoProcessor освобождены")