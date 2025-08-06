import os
import argparse
import yaml
import torch
import shutil
from ultralytics import YOLO
import logging

# Настройка логирования (только консоль, русский)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def check_gpu():
    """Проверка доступности GPU."""
    logger.info(f"🔥 Версия PyTorch: {torch.__version__}")
    cuda_v = torch.version.cuda or "none"
    cudnn_v = torch.backends.cudnn.version() or "none"
    logger.info(f"🔧 Скомпилировано с CUDA: {cuda_v}")
    logger.info(f"🔧 Версия cuDNN: {cudnn_v}")

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        logger.info(f"✅ Обнаружено {count} CUDA-устройств")
        for idx in range(count):
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_mem = props.total_memory / (1024 ** 3)
            logger.info(f"🎮 GPU {idx}: {name}, {total_mem:.1f} GB")
        return 'cuda:0'
    else:
        logger.warning("⚠️ torch.cuda.is_available() == False → GPU не обнаружен")
        logger.info("📝 Убедитесь, что PyTorch установлен с поддержкой CUDA")
        logger.info("📝 Проверьте установку драйвера NVIDIA и CUDA Toolkit")
        return 'cpu'

def create_default_yaml(dataset_path, class_name):
    """Создание конфигурации data.yaml."""
    default_yaml = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': [class_name]
    }
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(default_yaml, f)
    logger.info(f"✅ Создан файл {yaml_path} с конфигурацией:")
    logger.info(f"📋 train: train/images")
    logger.info(f"📋 val: val/images")
    logger.info(f"📋 test: test/images")
    logger.info(f"📋 nc: 1")
    logger.info(f"📋 names: [{class_name}]")
    return yaml_path

def update_yaml_paths(data_yaml_path, base_path):
    """Обновление путей в data.yaml."""
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Файл {data_yaml_path} пуст или имеет неверный формат")

        for split in ['train', 'val', 'test']:
            if split in data and data[split]:
                rel = data[split]
                abs_path = os.path.abspath(os.path.join(base_path, rel))
                if not os.path.exists(abs_path):
                    logger.warning(f"⚠️ Директория «{abs_path}» не найдена")
                data[split] = abs_path

        updated_yaml = os.path.join(base_path, 'updated_data.yaml')
        with open(updated_yaml, 'w') as f:
            yaml.safe_dump(data, f)

        logger.info(f"📋 Содержимое {updated_yaml}:")
        with open(updated_yaml, 'r') as f:
            logger.info(f.read())

        return updated_yaml

    except Exception as e:
        logger.error(f"🚨 Ошибка при обработке {data_yaml_path}: {e}")
        raise

def validate_dataset(dataset_path):
    """Валидация структуры датасета."""
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"🚨 Не найден {data_yaml}")

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    for split in ['train', 'val']:
        split_path = os.path.join(dataset_path, data[split])
        if not os.path.exists(split_path) or not os.listdir(split_path):
            raise ValueError(f"🚨 Директория {split_path} пуста или не существует")

def train_model(dataset_path, output_dir, class_name, epochs=50, img_size=640, batch_size=16):
    """Обучение YOLO модели."""
    try:
        validate_dataset(dataset_path)
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        updated_data_yaml = update_yaml_paths(data_yaml, dataset_path)
        device = check_gpu()
        model = YOLO(os.getenv('MODEL_PATH', 'models/yolo11m-seg.pt'))

        logger.info(f"🚀 Начало обучения на {dataset_path} (device={device})")
        results = model.train(
            data=updated_data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='temp',
            project='temp',
            task='detect',
            device=device,
            patience=10,
            save=True,
            exist_ok=True
        )

        best = os.path.join('temp', 'temp', 'weights', 'best.pt')
        if not os.path.exists(best):
            raise RuntimeError("🚨 Обучение завершено, но best.pt не найден")

        os.makedirs(output_dir, exist_ok=True)
        final = os.path.join(output_dir, "yolo11m-seg-mod.pt")
        os.replace(best, final)

        shutil.rmtree('temp', ignore_errors=True)
        logger.info(f"✅ Модель сохранена в {final}")
        return final

    except Exception as e:
        logger.error(f"🚨 Ошибка в train_model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Обучение YOLO модели на пользовательском датасете")
    parser.add_argument('--dataset_dir', type=str, default='../train/object', help="Путь к датасету")
    parser.add_argument('--output_dir', type=str, default=os.getenv('MODEL_DIR', 'models'), help="Путь для сохранения модели")
    parser.add_argument('--class_name', type=str, default='pig', help="Имя класса")
    parser.add_argument('--epochs', type=int, default=50, help="Количество эпох")
    parser.add_argument('--img_size', type=int, default=640, help="Размер кадра")
    parser.add_argument('--batch_size', type=int, default=16, help="Размер батча")
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, args.class_name)
    if not os.path.exists(dataset_path):
        logger.error(f"🚨 Папка датасета не найдена: {dataset_path}")
        return

    logger.info(f"🏷️ Датасет: {dataset_path} | класс: {args.class_name}")
    train_model(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        class_name=args.class_name,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()