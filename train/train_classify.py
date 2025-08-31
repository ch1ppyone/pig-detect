#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🏋️‍♂️ Конвертация YOLO-данных и обучение MobileNetV3Small на PyTorch для классификации поз свиней.
Поддержка лёгкой и расширенной аугментации.
"""

import os
import sys
import argparse
import cv2
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Добавляем корень проекта в sys.path для прямого запуска
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db import DatabaseManager

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Гиперпараметры
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.20
SEED = 42
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
MODEL_OUT = os.getenv('PT_MODEL_PATH', os.path.join(MODEL_DIR, "pig_model.pth"))
DB_PATH = os.getenv('DB_PATH', 'pig_states.db')
IMG_DIR = os.getenv('IMG_DIR', 'train/images')
LABEL_DIR = os.getenv('LABEL_DIR', 'train/labels')


def convert_yolo_to_tf(img_dir=IMG_DIR, label_dir=LABEL_DIR, out_dir=DATASET_DIR, db=None):
    """Конвертация YOLO-данных в формат для PyTorch (ImageFolder)."""
    logger.info("🔧 Начало конвертации YOLO-данных в формат для PyTorch")
    class_names = db.load_class_names()

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for cls in class_names:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        logger.info(f"📁 Создана папка для класса: {cls_dir}")

    processed, skipped = 0, 0

    for filename in os.listdir(img_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        name = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, f"{name}.txt")

        if not os.path.exists(label_path):
            logger.warning(f"⚠️ Нет метки: {label_path}")
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"⚠️ Не удалось загрузить изображение: {img_path}")
            skipped += 1
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            logger.warning(f"⚠️ Пустой .txt файл: {label_path}")
            skipped += 1
            continue

        for i, line in enumerate(lines):
            try:
                cls_id, x, y, bw, bh = map(float, line.split())
                cls_id = int(cls_id)
                if cls_id >= len(class_names):
                    logger.warning(f"⚠️ Неверный ID класса {cls_id} в {label_path}")
                    skipped += 1
                    continue
                cls_name = class_names[cls_id]
            except Exception as e:
                logger.error(f"🚨 Ошибка разбора строки в {label_path}: {e}")
                skipped += 1
                continue

            cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0 or (x2 - x1) < 5 or (y2 - y1) < 5:
                logger.warning(f"⚠️ Пустой/слишком маленький bbox в {label_path}")
                skipped += 1
                continue

            out_path = os.path.join(out_dir, cls_name, f"{name}_{i}.jpg")
            cv2.imwrite(out_path, roi)
            processed += 1

    logger.info(f"🎉 Конвертация завершена! Обработано: {processed}, пропущено: {skipped}")


def get_transforms(mode="light"):
    """Возвращает трансформации для train/val"""
    if mode == "light":
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:  # heavy
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )], p=0.7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return train_transform, val_transform


def train_model(db, aug_mode="light"):
    """Обучение MobileNetV3Small на PyTorch и сохранение модели."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Используемое устройство: {device}")

    train_transform, val_transform = get_transforms(mode=aug_mode)

    logger.info("📦 Загрузка датасета...")
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = db.load_class_names()
    num_classes = len(class_names)
    logger.info(f"🏷️ Классы: {class_names}")

    model = models.mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    logger.info("✅ Модель MobileNetV3Small загружена")

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    logger.info("🚀 Этап 1: обучение верхушки...")
    for param in model.features.parameters():
        param.requires_grad = False

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            acc = 100 * running_correct / total
            progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        logger.info(f"Эпоха {epoch+1}, Потери: {running_loss/len(train_loader.dataset):.4f}, Точность: {100*running_correct/total:.2f}%")

    logger.info("🚀 Этап 2: Fine-tuning последних слоёв...")
    for param in model.features[-3:].parameters():  # размораживаем хвост сети
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)

    for epoch in range(5):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/5 (Fine-tuning)", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            acc = 100 * running_correct / total
            progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        logger.info(f"Fine-tuning {epoch+1}, Потери: {running_loss/len(train_loader.dataset):.4f}, Точность: {100*running_correct/total:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    logger.info(f"✅ Модель сохранена в {MODEL_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Конвертация и обучение MobileNetV3Small на PyTorch для классификации поз свиней")
    parser.add_argument('--convert', action='store_true', help="Конвертировать YOLO-данные в формат PyTorch")
    parser.add_argument('--light-aug', action='store_true', help="Лёгкая аугментация")
    parser.add_argument('--heavy-aug', action='store_true', help="Сильная аугментация")
    args = parser.parse_args()

    db = DatabaseManager(db_path=DB_PATH)

    if args.convert:
        convert_yolo_to_tf(db=db)

    if args.heavy_aug:
        aug_mode = "heavy"
    else:
        aug_mode = "light"

    train_model(db, aug_mode=aug_mode)
    db.close()


if __name__ == '__main__':
    main()
