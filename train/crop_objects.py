#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np

def _read_img(p: Path):
    arr = np.fromfile(str(p), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _write_img(p: Path, img):
    ext = '.' + p.suffix.lstrip('.')
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(p))
    return True

def parse_yolo_polygon(line, img_width, img_height, target_class_id=0):
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    class_id = int(parts[0])
    if class_id != target_class_id:
        return None
    coords = [float(x) for x in parts[1:]]
    if len(coords) % 2 != 0:
        return None
    points = []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_width
        y = coords[i + 1] * img_height
        points.append([x, y])
    points = np.array(points)
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return (x_min, y_min, x_max, y_max)

def crop_objects_from_image(img_path, label_path, output_dir, base_name, target_class_id=0):
    try:
        if not img_path.exists():
            return 0, "missing_image"
        img = _read_img(img_path)
        if img is None:
            return 0, "corrupted_image"
        img_height, img_width = img.shape[:2]
        if not label_path.exists():
            return 0, "missing_label"
        cropped_count = 0
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            bbox = parse_yolo_polygon(line, img_width, img_height, target_class_id)
            if bbox is None:
                continue
            x_min, y_min, x_max, y_max = bbox
            if x_min >= x_max or y_min >= y_max:
                continue
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img_width, x_max + padding)
            y_max = min(img_height, y_max + padding)
            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue
            output_filename = f"{base_name}_crop_{i}.jpg"
            output_path = output_dir / output_filename
            if _write_img(output_path, cropped):
                cropped_count += 1
        return cropped_count, "success"
    except Exception as e:
        return 0, f"error: {e}"

def create_dataset(input_dir: Path, output_dir: Path, target_class_id=0):
    if not input_dir.exists():
        print(f"Ошибка: {input_dir} не существует!")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Вход: {input_dir}")
    print(f"Выход: {output_dir}")

    subsets = ["train", "val", "test"]
    total_images = 0
    total_cropped = 0
    error_stats = {
        "missing_image": 0,
        "corrupted_image": 0,
        "missing_label": 0,
        "success": 0,
        "errors": 0
    }

    for subset in subsets:
        img_dir = input_dir / subset / "images"
        label_dir = input_dir / subset / "labels"
        if not img_dir.exists() or not label_dir.exists():
            print(f"Пропускаем {subset} — нет папок")
            continue
        print(f"\nОбработка набора: {subset}")
        image_files = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"Найдено изображений: {len(image_files)}")
        for img_file in image_files:
            label_file = label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                base_name = f"{subset}_{img_file.stem}"
                cropped_count, status = crop_objects_from_image(img_file, label_file, output_dir, base_name, target_class_id)
                total_cropped += cropped_count
                total_images += 1
                if status == "success":
                    error_stats["success"] += 1
                elif status in error_stats:
                    error_stats[status] += 1
                else:
                    error_stats["errors"] += 1
                if total_images % 100 == 0:
                    print(f"Обработано: {total_images}, вырезано: {total_cropped}")

    print(f"\n✅ Готово")
    print(f"Обработано изображений: {total_images}")
    print(f"Всего кропов: {total_cropped}")
    print(f"\nСтатистика:")
    for k, v in error_stats.items():
        print(f"  {k}: {v}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python crop_objects.py <входная_папка_класса> <выходная_папка> [class_id]")
        sys.exit(1)
    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    class_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    create_dataset(input_dir, output_dir, class_id)
