import unittest
import cv2
import numpy as np
from dotenv import load_dotenv
from app.detector import Detector

load_dotenv()

class TestDetector(unittest.TestCase):
    def setUp(self):
        """Подготовка тестового детектора."""
        self.model_path = os.getenv('MODEL_PATH', "models/yolo11m-seg-mod.pt")
        self.device = "cpu"
        self.tracker_config = os.getenv('TRACKER_CONFIG', "models/bytetrack.yaml")
        self.detector = Detector(self.model_path, self.device, self.tracker_config)

    def test_init_detector(self):
        """Тест инициализации детектора."""
        self.assertEqual(self.detector.model_path, self.model_path)
        self.assertEqual(self.detector.device, self.device)
        self.assertFalse(self.detector.tracker_available)

    def test_detect_and_track(self):
        """Тест детекции на пустом кадре."""
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = self.detector.detect_and_track(frame)
        self.assertIsNotNone(results)  # Предполагаем, что YOLO возвращает объект результатов

if __name__ == '__main__':
    unittest.main()