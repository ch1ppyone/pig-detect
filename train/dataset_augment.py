#!/usr/bin/env python3
"""
Максимально быстрая аугментация датасета с многопоточностью.
Оптимизирован для скорости обработки.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def fast_augmentations():
    """Быстрые аугментации без сложных вычислений."""
    return [
        {
            'name': 'flip_h',
            'func': lambda img: cv2.flip(img, 1),
            'bbox_func': lambda bboxes: [[1-bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes]
        },
        {
            'name': 'bright',
            'func': lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=15),
            'bbox_func': lambda bboxes: bboxes
        },
        {
            'name': 'dark',
            'func': lambda img: cv2.convertScaleAbs(img, alpha=0.8, beta=-15),
            'bbox_func': lambda bboxes: bboxes
        },
        {
            'name': 'contrast',
            'func': lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=0),
            'bbox_func': lambda bboxes: bboxes
        },
        {
            'name': 'blur',
            'func': lambda img: cv2.blur(img, (3, 3)),  # Быстрее чем GaussianBlur
            'bbox_func': lambda bboxes: bboxes
        }
    ]

def read_yolo_fast(label_path):
    """Быстрое чтение YOLO разметки."""
    if not os.path.exists(label_path):
        return []
    
    try:
        with open(label_path, 'r') as f:
            return [line.strip().split() for line in f if line.strip()]
    except:
        return []

def write_yolo_fast(label_path, labels):
    """Быстрая запись YOLO разметки."""
    try:
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    except:
        pass

def process_single_image(args):
    """Обрабатываем одно изображение (для многопоточности)."""
    img_path, label_path, output_images_dir, output_labels_dir, augmentations, aug_count = args
    
    try:
        # Читаем изображение
        image = cv2.imread(str(img_path))
        if image is None:
            return 0
        
        # Читаем разметку
        labels = read_yolo_fast(label_path)
        
        base_name = img_path.stem
        processed = 0
        
        # Копируем оригинал
        shutil.copy2(img_path, output_images_dir / img_path.name)
        if labels:
            write_yolo_fast(output_labels_dir / (base_name + '.txt'), labels)
        else:
            # Пустая разметка
            (output_labels_dir / (base_name + '.txt')).touch()
        processed += 1
        
        # Создаем аугментированные версии
        selected_augs = random.sample(augmentations, min(aug_count, len(augmentations)))
        
        for i, aug in enumerate(selected_augs):
            try:
                # Применяем аугментацию
                aug_image = aug['func'](image.copy())
                
                # Трансформируем боксы
                if labels:
                    bboxes = [[float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
                    aug_bboxes = aug['bbox_func'](bboxes)
                    aug_labels = [[labels[j][0]] + list(map(str, bbox)) for j, bbox in enumerate(aug_bboxes)]
                else:
                    aug_labels = []
                
                # Сохраняем
                aug_name = f"{base_name}_{aug['name']}"
                cv2.imwrite(str(output_images_dir / f"{aug_name}.jpg"), aug_image)
                write_yolo_fast(output_labels_dir / f"{aug_name}.txt", aug_labels)
                processed += 1
                
            except Exception as e:
                continue  # Пропускаем ошибочные аугментации
        
        return processed
        
    except Exception as e:
        return 0

def fast_augment_dataset(source_dir, output_dir, augment_count=2, num_workers=None):
    """
    Быстрая аугментация с многопоточностью.
    
    Args:
        source_dir: Исходный датасет
        output_dir: Выходной датасет
        augment_count: Количество аугментаций на изображение
        num_workers: Количество потоков (None = автоопределение)
    """
    
    source = Path(source_dir)
    output = Path(output_dir)
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Максимум 8 потоков
    
    print(f"⚡ Быстрая аугментация с {num_workers} потоками")
    
    # Удаляем старый датасет
    if output.exists():
        shutil.rmtree(output)
    
    # Создаем структуру
    for split in ['train', 'val', 'test']:
        (output / split / 'images').mkdir(parents=True, exist_ok=True)
        (output / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Получаем аугментации
    augmentations = fast_augmentations()
    
    total_processed = 0
    
    # Обрабатываем каждый split
    for split in ['train', 'val', 'test']:
        source_images = source / split / "images"
        source_labels = source / split / "labels"
        
        if not source_images.exists():
            continue
        
        # Получаем список файлов
        image_files = [f for f in source_images.iterdir() 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        if not image_files:
            continue
        
        print(f"\n🎨 Аугментация {split}: {len(image_files)} изображений")
        
        output_images_dir = output / split / "images"
        output_labels_dir = output / split / "labels"
        
        # Подготавливаем аргументы для многопоточности
        process_args = []
        for img_file in image_files:
            label_file = source_labels / (img_file.stem + '.txt')
            process_args.append((
                img_file, label_file, output_images_dir, output_labels_dir, 
                augmentations, augment_count
            ))
        
        # Многопоточная обработка
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_image, process_args),
                total=len(process_args),
                desc=f"Обработка {split}"
            ))
        
        split_processed = sum(results)
        total_processed += split_processed
        print(f"   ✅ {split}: обработано {split_processed} файлов")
    
    return total_processed

def create_fast_yaml(output_dir, total_count):
    """Быстрое создание dataset.yaml."""
    output = Path(output_dir)
    
    config = {
        'path': str(output.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': ['pig'],
        'nc': 1,
        'augmented': True,
        'total_images': total_count
    }
    
    yaml_path = output / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path

def main():
    """Главная функция быстрой аугментации."""
    parser = argparse.ArgumentParser(description="Быстрая аугментация датасета")
    parser.add_argument("--source", type=str, default="train/object/pig", help="Исходный датасет")
    parser.add_argument("--output", type=str, default="train/object/pig_fast_aug", help="Выходной датасет")
    parser.add_argument("--count", type=int, default=2, help="Аугментаций на изображение")
    parser.add_argument("--workers", type=int, default=None, help="Количество потоков")
    
    args = parser.parse_args()
    
    print("⚡ БЫСТРАЯ АУГМЕНТАЦИЯ ДАТАСЕТА")
    print("=" * 50)
    
    if not os.path.exists(args.source):
        print(f"❌ Датасет не найден: {args.source}")
        return
    
    start_time = cv2.getTickCount()
    
    try:
        # Быстрая аугментация
        total_processed = fast_augment_dataset(
            args.source, 
            args.output, 
            args.count,
            args.workers
        )
        
        # Создаем yaml
        yaml_path = create_fast_yaml(args.output, total_processed)
        
        # Вычисляем время
        end_time = cv2.getTickCount()
        elapsed = (end_time - start_time) / cv2.getTickFrequency()
        
        print(f"\n⚡ БЫСТРАЯ АУГМЕНТАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 50)
        print(f"📊 Обработано файлов: {total_processed}")
        print(f"⏱️ Время выполнения: {elapsed:.1f} секунд")
        print(f"🚀 Скорость: {total_processed/elapsed:.1f} файлов/сек")
        print(f"📁 Результат: {args.output}")
        
        print(f"\n🎯 КОМАНДА ДЛЯ ОБУЧЕНИЯ:")
        print(f"python train/train_yolo.py --dataset_dir {args.output} --class_name pig --epochs 100")
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


