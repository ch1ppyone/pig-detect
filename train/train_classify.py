#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🏋️‍♂️ Обучение MobileNetV3Small на датасете поз свиней (ImageFolder).
Добавлен EarlyStopping и более агрессивная аугментация.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# дефолтные гиперпараметры
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.2
SEED = 42
MODEL_OUT = "pig_model_best.pth"


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def step(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return True  # новое лучшее значение
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def get_transforms(mode="light", image_size=IMAGE_SIZE):
    if mode == "light":
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:  # heavy
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.7, 1.3), shear=15),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    return train_transform, val_transform


def train_model(dataset_dir, aug_mode, device, epochs, batch_size, image_size, out_path):
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"🔧 Устройство: {device}")

    train_transform, val_transform = get_transforms(mode=aug_mode, image_size=image_size)

    print("📦 Загружаем датасет...")
    full_dataset = datasets.ImageFolder(dataset_dir)
    num_classes = len(full_dataset.classes)
    print(f"🏷️ Классы: {full_dataset.classes}")

    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Sequential(
        nn.Linear(model.classifier[3].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    early_stopping = EarlyStopping(patience=5)

    best_path = out_path
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = 100 * running_correct / total
            progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * running_correct / total
        print(f"📊 Train {epoch+1}: loss {train_loss:.4f}, acc {train_acc:.2f}%")

        # валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        print(f"🧪 Val {epoch+1}: loss {val_loss:.4f}, acc {val_acc:.2f}%")

        # сохранение лучшей модели
        if early_stopping.step(val_acc):
            torch.save(model.state_dict(), best_path)
            print(f"💾 Лучшая модель сохранена (val_acc={val_acc:.2f}%)")

        if early_stopping.early_stop:
            print("⏹️ Early stopping: точность не улучшается")
            break

    print(f"✅ Обучение завершено. Лучшая точность: {early_stopping.best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Обучение MobileNetV3Small для классификации поз свиней")
    parser.add_argument("--dataset", type=str, default="dataset", help="Путь к датасету (ImageFolder)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Размер батча")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Устройство")
    parser.add_argument("--imgsz", type=int, default=224, help="Размер изображения")
    parser.add_argument("--heavy-aug", action="store_true", help="Сильная аугментация")
    parser.add_argument("--out", type=str, default=MODEL_OUT, help="Файл для сохранения модели")
    args = parser.parse_args()

    aug_mode = "heavy" if args.heavy_aug else "light"
    train_model(
        dataset_dir=args.dataset,
        aug_mode=aug_mode,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=(args.imgsz, args.imgsz),
        out_path=args.out
    )


if __name__ == "__main__":
    main()
