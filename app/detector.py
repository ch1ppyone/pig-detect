from ultralytics import YOLO
from .logging import logger


class Detector:
    def __init__(self, model_path, device, tracker_config="bytetrack.yaml"):
        """
        Инициализация детектора YOLO.

        Args:
            model_path (str): Путь к модели YOLO.
            device (str): Устройство (cuda/cpu).
            tracker_config (str): Конфигурация трекера.
        """
        self.model_path = model_path
        self.device = device
        self.tracker_config = tracker_config
        self.tracker_available = False  # Отключено из-за ошибки fuse_score

        logger.info(f"🔧 Инициализация детектора: model={model_path}, device={device}, tracker={tracker_config}")
        try:
            self.detector = YOLO(model_path)
            self.detector.to(device)
            logger.info(f"✅ YOLO-модель {'на GPU' if device == 'cuda' else 'на CPU'}")
        except Exception as e:
            logger.error(f"🚨 Ошибка инициализации YOLO: {str(e)}")
            raise

    def detect_and_track(self, frame):
        """
        Детекция и трекинг объектов на кадре.

        Args:
            frame: Входной кадр (numpy array).

        Returns:
            Результаты детекции/трекинга.
        """
        try:
            results = self.detector.predict(frame, conf=0.5, verbose=True)
            num_objects = len(results[0].boxes) if results and results[0].boxes else 0
            logger.info(f"🔍 Детекция выполнена, найдено {num_objects} объектов")
            return results
        except Exception as e:
            logger.error(f"🚨 Ошибка в detect_and_track: {str(e)}")
            return None