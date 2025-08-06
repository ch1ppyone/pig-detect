#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🏋️‍♂️ Конвертация YOLO-данных и обучение MobileNetV3Small на PyTorch для классификации поз свиней.
Сохраняет модель PyTorch (pig_model.pth) для инференса.
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

# Добавляем корень проекта в sys.path для прямого запуска
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db import DatabaseManager
import logging

# Настройка логирования (только консоль, русский)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Гиперпараметры
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.20
SEED = 42
DATASET_DIR = "../train/dataset/"
MODEL_DIR = os.getenv('MODEL_DIR', '../models')
MODEL_OUT = os.getenv('PT_MODEL_PATH', os.path.join(MODEL_DIR, "pig_model.pth"))
DB_PATH = os.getenv('DB_PATH', 'pig_states.db')
IMG_DIR = os.getenv('IMG_DIR', 'train/object/pig_pose/train/images')
LABEL_DIR = os.getenv('LABEL_DIR', 'train/object/pig_pose/train/labels')


def convert_yolo_to_tf(img_dir=IMG_DIR, label_dir=LABEL_DIR, out_dir=DATASET_DIR, db=None):
    """Конвертация YOLO-данных в формат для PyTorch."""
    logger.info("🔧 Начало конвертации YOLO-данных в формат для PyTorch")
    class_names = db.load_class_names()

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for cls in class_names:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        logger.info(f"📁 Создана папка для класса: {cls_dir}")

    processed = 0
    skipped = 0

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

        try:
            cls_id, x, y, bw, bh = map(float, lines[0].split())
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
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0 or (x2 - x1) < 5 or (y2 - y1) < 5:
            logger.warning(f"⚠️ Пустой/слишком маленький bbox в {label_path}")
            skipped += 1
            continue

        out_path = os.path.join(out_dir, cls_name, f"{name}.jpg")
        cv2.imwrite(out_path, roi)
        logger.info(f"✅ {filename} → {cls_name} (сохранено в {out_path})")
        processed += 1

    logger.info(f"🎉 Конвертация завершена! Обработано: {processed}, пропущено: {skipped}")


def train_model(db):
    """Обучение MobileNetV3Small на PyTorch и сохранение модели."""
    # Проверка и настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Используемое устройство: {device}")

    logger.info("📦 Загрузка датасета...")
    try:
        # Аугментация и предобработка
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Разделение на train/val
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
    except Exception as e:
        logger.error(f"🚨 Ошибка загрузки датасета: {e}")
        raise

    class_names = db.load_class_names()
    num_classes = len(class_names)
    logger.info(f"🏷️ Классы: {class_names}")

    # Инициализация модели
    try:
        model = models.mobilenet_v3_small(weights='DEFAULT')
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model = model.to(device)
        logger.info("✅ Модель MobileNetV3Small загружена")
    except Exception as e:
        logger.error(f"🚨 Ошибка инициализации модели: {e}")
        raise

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Этап 1: Обучение верхушки
    logger.info("🚀 Этап 1: обучение верхушки...")
    model.classifier.train()
    for param in model.features.parameters():
        param.requires_grad = False

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * running_correct / total
        logger.info(f"Эпоха {epoch+1}/{EPOCHS}, Потери: {epoch_loss:.4f}, Точность: {epoch_acc:.2f}%")

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(val_loader, desc="Валидация", unit="batch")
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                acc = 100 * correct / total
                progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        logger.info(f"Валидация: Потери: {val_loss:.4f}, Точность: {val_acc:.2f}%")

    # Этап 2: Fine-tuning
    logger.info("🚀 Этап 2: тонкая настройка последних слоёв...")
    model.features[-20:].train()
    for param in model.features[:-20].parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * running_correct / total
        logger.info(f"Эпоха {epoch+1}/5, Потери: {epoch_loss:.4f}, Точность: {epoch_acc:.2f}%")

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(val_loader, desc="Валидация (Fine-tuning)", unit="batch")
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                acc = 100 * correct / total
                progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        logger.info(f"Валидация: Потери: {val_loss:.4f}, Точность: {val_acc:.2f}%")

    # Сохранение модели
    logger.info("💾 Сохранение модели PyTorch...")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUT)
        logger.info(f"✅ Модель сохранена в {MODEL_OUT}")
    except Exception as e:
        logger.error(f"🚨 Ошибка сохранения модели: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Конвертация и обучение MobileNetV3Small на PyTorch для классификации поз свиней")
    parser.add_argument('--convert', action='store_true', help="Конвертировать YOLO-данные в формат PyTorch")
    args = parser.parse_args()

    db = DatabaseManager(db_path=DB_PATH)

    if args.convert:
        convert_yolo_to_tf(db=db)

    train_model(db)
    db.close()


if __name__ == '__main__':
    main()