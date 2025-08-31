#!/usr/bin/env python3
"""
Обучение YOLO: можно стартовать с нуля или продолжить с last/best.
Автоматически определяет: нужно ли resume, или новый запуск.
"""

import os
import argparse
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def check_gpu():
    print(f"🧠 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA доступна ({torch.cuda.get_device_name(0)})")
        return 0
    else:
        print("⚠️ GPU не найден, обучение на CPU")
        return "cpu"

def fast_train_model(dataset_path, epochs=50, img_size=640, batch_size=16, weights=None, resume=False):
    device = check_gpu()

    # Загружаем модель
    if weights:
        print(f"🔄 Используем веса: {weights}")
        model = YOLO(weights)
    else:
        print("🚀 Стартуем с yolo11n.pt")
        model = YOLO("yolo11n.pt")

    try:
        if resume:
            print("⏯ Попытка возобновить обучение...")
            results = model.train(resume=True, epochs=epochs, device=device)
        else:
            print("▶ Начинаем новое обучение...")
            results = model.train(
                data=str(dataset_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                workers=8,
                cache=False,
                amp=True,
                patience=10,
                save_period=10,
                val=True,
                plots=False,
                verbose=True,
                single_cls=True,
                rect=True,
                cos_lr=True,
                close_mosaic=10,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0
            )
        return results
    except AssertionError as e:
        if "nothing to resume" in str(e):
            print("⚠️ Обучение завершено, продолжаем как новый запуск...")
            results = model.train(
                data=str(dataset_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                workers=8,
                amp=True,
                single_cls=True,
                cos_lr=True
            )
            return results
        else:
            raise

def main():
    parser = argparse.ArgumentParser(description="Обучение YOLO")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Путь к датасету")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--img_size", type=int, default=640, help="Размер изображения")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча")
    parser.add_argument("--weights", type=str, default=None, help="Путь к весам (last.pt или best.pt)")
    parser.add_argument("--resume", action="store_true", help="Попробовать возобновить обучение")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    yaml_file = dataset_path / "dataset.yaml"
    if not yaml_file.exists():
        print(f"❌ dataset.yaml не найден: {yaml_file}")
        return

    fast_train_model(yaml_file, args.epochs, args.img_size, args.batch_size, args.weights, args.resume)

if __name__ == "__main__":
    main()
