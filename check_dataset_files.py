#!/usr/bin/env python3
"""
Скрипт для проверки содержимого папок датасета.
"""

import sys
import os
from pathlib import Path

# Добавляем путь к приложению
sys.path.insert(0, str(Path(__file__).parent))

def check_dataset_structure():
    """Проверка структуры датасета."""
    print("📂 Проверка структуры датасета")
    print("=" * 50)
    
    try:
        from app.config import config_manager
        from app.routes import StateManager
        
        # Получаем список состояний из БД
        db = StateManager.get_db()
        states = db.get_states()
        
        print(f"🏷️ Состояний в БД: {len(states)}")
        for code, desc in states.items():
            print(f"   • {code}: {desc}")
        
        print("\n📁 Проверка папок датасета:")
        
        dataset_root = os.path.join("train", "dataset")
        if not os.path.exists(dataset_root):
            print(f"❌ Корневая папка датасета не существует: {dataset_root}")
            return False
        
        print(f"✅ Корневая папка датасета: {dataset_root}")
        
        # Проверяем каждое состояние
        for state_code in states.keys():
            state_dir = os.path.join(dataset_root, state_code)
            print(f"\n📋 Состояние: {state_code}")
            print(f"   Папка: {state_dir}")
            
            if not os.path.exists(state_dir):
                print("   ❌ Папка не существует")
                continue
            
            # Получаем список файлов
            try:
                all_files = os.listdir(state_dir)
                print(f"   📊 Всего файлов: {len(all_files)}")
                
                if len(all_files) == 0:
                    print("   ⚠️ Папка пустая")
                    continue
                
                # Анализируем типы файлов
                video_files = []
                image_files = []
                other_files = []
                
                video_extensions = config_manager.config.allowed_video_extensions
                image_extensions = config_manager.config.allowed_image_extensions
                
                for filename in all_files:
                    extension = filename.lower().split('.')[-1] if '.' in filename else ''
                    
                    if extension in video_extensions:
                        video_files.append(filename)
                    elif extension in image_extensions:
                        image_files.append(filename)
                    else:
                        other_files.append(filename)
                
                print(f"   🎬 Видео файлов: {len(video_files)}")
                for video in video_files[:3]:  # Показываем первые 3
                    print(f"      • {video}")
                if len(video_files) > 3:
                    print(f"      ... и еще {len(video_files) - 3}")
                
                print(f"   🖼️ Изображений: {len(image_files)}")
                for image in image_files[:3]:  # Показываем первые 3
                    print(f"      • {image}")
                if len(image_files) > 3:
                    print(f"      ... и еще {len(image_files) - 3}")
                
                if other_files:
                    print(f"   ❓ Других файлов: {len(other_files)}")
                    for other in other_files[:3]:
                        print(f"      • {other}")
                
            except Exception as e:
                print(f"   ❌ Ошибка чтения папки: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки датасета: {str(e)}")
        import traceback
        print(f"Трассировка: {traceback.format_exc()}")
        return False

def test_api_endpoint():
    """Тест API эндпоинта get_videos."""
    print("\n🌐 Тестирование API эндпоинта")
    print("=" * 50)
    
    try:
        from app.routes import StateManager
        from app.config import config_manager
        
        # Получаем список состояний
        db = StateManager.get_db()
        states = list(db.get_states().keys())
        
        if not states:
            print("❌ Нет состояний для тестирования")
            return False
        
        # Тестируем первое состояние
        test_state = states[0]
        print(f"🧪 Тестируем состояние: {test_state}")
        
        # Симулируем логику из get_videos
        state_dir = os.path.join("train", "dataset", test_state)
        print(f"📁 Путь: {state_dir}")
        
        if not os.path.exists(state_dir):
            print("❌ Папка не существует")
            return False
        
        all_files = []
        all_files_in_dir = os.listdir(state_dir)
        print(f"📋 Всего файлов в папке: {len(all_files_in_dir)}")
        
        video_count = 0
        image_count = 0
        other_count = 0
        
        for filename in all_files_in_dir:
            extension = filename.lower().split('.')[-1] if '.' in filename else ''
            
            if extension in config_manager.config.allowed_video_extensions:
                all_files.append(filename)
                video_count += 1
            elif extension in config_manager.config.allowed_image_extensions:
                all_files.append(filename)
                image_count += 1
            else:
                other_count += 1
        
        print(f"📊 Результат фильтрации:")
        print(f"   🎬 Видео: {video_count}")
        print(f"   🖼️ Изображения: {image_count}")
        print(f"   ❓ Другие: {other_count}")
        print(f"   ✅ Всего медиа файлов: {len(all_files)}")
        
        # Проверяем пагинацию
        per_page = 10
        total_pages = (len(all_files) + per_page - 1) // per_page
        print(f"📄 Страниц для пагинации: {total_pages}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования API: {str(e)}")
        import traceback
        print(f"Трассировка: {traceback.format_exc()}")
        return False

def main():
    """Главная функция."""
    print("🔍 Диагностика датасета")
    print("=" * 60)
    
    structure_ok = check_dataset_structure()
    api_ok = test_api_endpoint()
    
    print("\n" + "=" * 60)
    if structure_ok and api_ok:
        print("🎉 Все проверки пройдены!")
        print("💡 Если в интерфейсе показывается '0 файлов', проверьте:")
        print("   1. Логи сервера при запросе /get_videos/")
        print("   2. Консоль браузера на ошибки JavaScript")
        print("   3. Правильность путей к файлам")
    else:
        print("❌ Обнаружены проблемы")
        print("💡 Рекомендации:")
        print("   1. Убедитесь, что папки train/dataset/<state_code> существуют")
        print("   2. Добавьте файлы в соответствующие папки")
        print("   3. Проверьте права доступа к файлам")

if __name__ == "__main__":
    main()
