import cv2
import os
import argparse
import torch
import numpy as np
from app import create_app
from app.config import logger, check_hardware, create_video_processor, VIDEO_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Система мониторинга поведения свиней")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Устройство для работы модели (cuda или cpu)")
    parser.add_argument("--model", type=str, default=os.getenv('MODEL_PATH', "models/yolo11m-seg-mod.pt"), help="Путь к модели YOLO")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help="Путь к видеофайлу")
    parser.add_argument("--pt-model", type=str, default=os.getenv('PT_MODEL_PATH', "models/pig_model.pth"), help="Путь к модели PyTorch")
    parser.add_argument("--tracker-config", type=str, default=os.getenv('TRACKER_CONFIG', "models/bytetrack.yaml"), help="Путь к конфигурации трекера")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Проверка оборудования
    device = check_hardware()

    # Проверка устройства
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA недоступен, переход на CPU")
        device = "cpu"

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("✅ Запуск на CPU")

    # Инициализация VideoProcessor
    try:
        processor = create_video_processor(args.video)
        logger.info("✅ VideoProcessor успешно инициализирован")
    except Exception as e:
        logger.error(f"🚨 Ошибка при инициализации процессора: {str(e)}")
        raise

    app = create_app(processor)
    app.run(host="0.0.0.0", port=9001)