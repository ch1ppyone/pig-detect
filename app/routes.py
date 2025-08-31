"""
Маршруты Flask для системы мониторинга свиней.
Организованы по функциональности с использованием классов.
"""

import os
import base64
import threading
import subprocess
import shlex
import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import cv2
from flask import render_template, jsonify, request, send_from_directory, Response
from flask_login import login_required

from flask import current_app
from . import socketio, frame_queue, processors, background_tasks
from .config import config_manager, SecurityUtils
from .logging import logger


class StreamManager:
    """Менеджер видеопотоков."""
    
    @staticmethod
    def frame_process_thread(processor_id: str) -> None:
        """Поток обработки кадров."""
        proc = processors.get(processor_id)
        if not proc:
            logger.error(f"🚨 Процессор {processor_id} не найден")
            return
        
        try:
            logger.info(f"🎬 Запущен поток обработки кадров для процессора {processor_id}")
            frame_count = 0
            while proc.running:
                frame = proc.next_frame()
                if frame is None:
                    logger.debug("⚠️ Получен пустой кадр")
                    socketio.sleep(0.1)
                    continue
                    
                if not isinstance(frame, np.ndarray):
                    logger.warning("⚠️ Неверный тип кадра")
                    continue
                    
                ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    logger.warning("⚠️ Не удалось закодировать кадр")
                    continue
                    
                data = base64.b64encode(buf).decode('utf-8')
                if frame_queue.full():
                    frame_queue.get()
                # Отправляем кадр всем подключенным клиентам
                for client_id in background_tasks.keys():
                    frame_queue.put((client_id, data))
                
                frame_count += 1
                if frame_count % 30 == 0:  # Логируем каждые 30 кадров
                    logger.debug(f"📊 Обработано {frame_count} кадров, подключенных клиентов: {len(background_tasks)}")
                
                socketio.sleep(proc.delay)
                
        except Exception as e:
            logger.error(f"🚨 Ошибка в потоке обработки кадров: {str(e)}")
            # Уведомляем всех клиентов об ошибке
            for client_id in background_tasks.keys():
                socketio.emit('error', {'message': f'Ошибка обработки кадра: {str(e)}'}, to=client_id)
    
    @staticmethod
    def frame_emit_thread() -> None:
        """Поток отправки кадров."""
        logger.info("📡 Запущен поток отправки кадров")
        emit_count = 0
        while True:
            if not frame_queue.empty():
                client_id, data = frame_queue.get()
                try:
                    socketio.emit('video_frame', {'data': data}, to=client_id)
                    emit_count += 1
                    if emit_count % 30 == 0:  # Логируем каждые 30 отправок
                        logger.debug(f"📤 Отправлено {emit_count} кадров клиенту {client_id}")
                except Exception as e:
                    logger.error(f"🚨 Ошибка отправки кадра клиенту {client_id}: {str(e)}")
            socketio.sleep(0.05)
    
    @staticmethod
    def data_emit_thread(processor_id: str) -> None:
        """Поток отправки логов."""
        proc = processors.get(processor_id)
        if not proc:
            logger.error(f"🚨 Процессор {processor_id} не найден")
            return
            
        try:
            while proc.running:
                logs = proc.logs[-100:]
                # Отправляем логи всем подключенным клиентам
                for client_id in background_tasks.keys():
                    socketio.emit('log_update', {'logs': logs[:10]}, to=client_id)
                socketio.sleep(2)
        except Exception as e:
            logger.error(f"🚨 Ошибка отправки логов: {str(e)}")
            # Уведомляем всех клиентов об ошибке
            for client_id in background_tasks.keys():
                socketio.emit('error', {'message': f'Ошибка отправки логов: {str(e)}'}, to=client_id)


class FileManager:
    """Менеджер файлов."""
    
    @staticmethod
    def get_source_files() -> List[Dict[str, Any]]:
        """Получение списка файлов из папки uploads."""
        uploads_dir = config_manager.config.upload_dir
        if not os.path.exists(uploads_dir):
            return []
        
        files = []
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            if not os.path.isfile(file_path):
                continue
                
            # Определяем тип файла
            extension = filename.lower().split('.')[-1]
            if extension in config_manager.config.allowed_video_extensions:
                file_type = '🎬 Видео'
                type_label = 'Видео'
            elif extension in config_manager.config.allowed_image_extensions:
                file_type = '🖼️ Изображение'
                type_label = 'Фото'
            else:
                continue  # Пропускаем неподдерживаемые файлы
            
            # Получаем размер файла
            try:
                size_bytes = os.path.getsize(file_path)
                size = FileManager._format_file_size(size_bytes)
            except Exception:
                size = "N/A"
            
            files.append({
                'name': filename,
                'type': file_type,
                'type_label': type_label,
                'size': size
            })
        
        return sorted(files, key=lambda x: x['name'])
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Форматирование размера файла."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    @staticmethod
    def get_file_info(filename: str) -> Dict[str, Any]:
        """Получение информации о файле."""
        file_path = os.path.join(config_manager.config.upload_dir, filename)
        
        # Определяем тип файла
        extension = filename.lower().split('.')[-1]
        if extension in config_manager.config.allowed_video_extensions:
            file_type = 'video'
            file_type_label = '🎬 Видео'
        elif extension in config_manager.config.allowed_image_extensions:
            file_type = 'image'
            file_type_label = '🖼️ Изображение'
        else:
            file_type = 'unknown'
            file_type_label = '❓ Неизвестный тип'
        
        # Получаем информацию о файле
        try:
            stat = os.stat(file_path)
            file_size = FileManager._format_file_size(stat.st_size)
            modified_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%d.%m.%Y %H:%M')
        except Exception as e:
            logger.error(f"🚨 Ошибка получения информации о файле {filename}: {str(e)}")
            file_size = "N/A"
            modified_time = "N/A"
        
        return {
            'filename': filename,
            'file_type': file_type,
            'file_type_label': file_type_label,
            'file_extension': extension,
            'file_size': file_size,
            'modified_time': modified_time
        }


class StateManager:
    """Менеджер состояний."""
    
    # Кеш для базы данных
    _db_cache = None
    _db_lock = threading.RLock()
    
    @staticmethod
    def get_db():
        """Получение кешированного соединения с БД."""
        with StateManager._db_lock:
            if StateManager._db_cache is None:
                from .db import DatabaseManager
                StateManager._db_cache = DatabaseManager(config_manager.config.db_path)
                logger.info("✅ Создано singleton соединение с БД")
            return StateManager._db_cache
    
    @staticmethod
    def get_processor_or_create(client_id: str = "default"):
        """Получение или создание процессора."""
        processor = processors.get(client_id)
        if not processor:
            try:
                processor = config_manager.create_video_processor()
                processors[client_id] = processor
                logger.info(f"✅ Создан процессор для клиента {client_id}")
            except Exception as e:
                logger.error(f"🚨 Ошибка создания процессора: {str(e)}")
                raise
        return processor
    
    @staticmethod
    def handle_training():
        """Обработка запроса на обучение модели."""
        def runner():
            try:
                proc = StateManager.get_processor_or_create()
                states = proc.db.get_states()
                
                # Проверяем наличие данных
                missing = [
                    s for s in states
                    if not os.path.isdir(os.path.join("train", "dataset", s))
                    or not os.listdir(os.path.join("train", "dataset", s))
                ]
                
                if missing:
                    socketio.emit('training_warning', 
                                {'message': f'Нет данных для классов: {",".join(missing)}'})
                    return
                
                # Запуск обучения
                cmd = "python train/train_classify.py"
                process = subprocess.Popen(
                    shlex.split(cmd), 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
                
                for line in process.stdout:
                    socketio.emit('training_log', {'log': line.strip()})
                
                code = process.wait()
                status = 'OK' if code == 0 else 'ERR'
                socketio.emit('training_complete', {'message': status})
                
                # Пересоздаем процессор с новой моделью
                proc.release()
                processors["default"] = config_manager.create_video_processor(proc.video_path)
                
            except Exception as e:
                logger.error(f"🚨 Ошибка обучения: {str(e)}")
                socketio.emit('training_warning', {'message': f'Ошибка: {str(e)}'})
        
        threading.Thread(target=runner, daemon=True).start()


# === МАРШРУТЫ ===

def register_routes(app):
    """Регистрация маршрутов для экземпляра приложения."""
    
    @app.route("/")
    @login_required
    def index():
        """Главная страница."""
        try:
            files = FileManager.get_source_files()
            vids = [f for f in files if 'Видео' in f['type_label']]
            imgs = [f for f in files if 'Фото' in f['type_label']]
            sources = [(f['name'], f['type_label']) for f in vids + imgs]
        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить список источников: {str(e)}")
            sources = []

        # Получаем текущий источник
        default = processors.get("default")
        current_source = "webcam"  # по умолчанию
        
        if default:
            try:
                current_source = os.path.basename(default.video_path)
            except Exception:
                pass

        return render_template("index.html", sources=sources, current_source=current_source)


    @app.route("/get_source_files")
    @login_required
    def get_source_files():
        """API для получения списка файлов."""
        try:
            logger.info("📁 Запрос списка файлов из uploads")
            files = FileManager.get_source_files()
            logger.info(f"📁 Найдено {len(files)} файлов")
            return jsonify(files=files)
        except Exception as e:
            logger.error(f"🚨 Ошибка получения списка файлов: {str(e)}")
            return jsonify(error="Ошибка получения списка файлов"), 500


    @app.route('/preview/<filename>')
    @login_required
    def preview_file(filename: str):
        """Страница предпросмотра файла."""
        logger.info(f"📋 Запрос предпросмотра файла: {filename}")
        
        if not SecurityUtils.is_safe_filename(filename):
            logger.error(f"🚨 Недопустимое имя файла: {filename}")
            return "Недопустимое имя файла", 400
        
        file_path = os.path.join(config_manager.config.upload_dir, filename)
        logger.info(f"📂 Проверяем путь к файлу: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"🚨 Файл не найден: {file_path}")
            return "Файл не найден", 404
        
        logger.info(f"✅ Рендерим шаблон preview.html для файла: {filename}")
        
        try:
            file_info = FileManager.get_file_info(filename)
            return render_template('preview.html', **file_info)
        except Exception as e:
            logger.error(f"🚨 Ошибка рендеринга шаблона: {str(e)}")
            return f"Ошибка рендеринга шаблона: {str(e)}", 500


    @app.route('/uploads/<filename>')
    @login_required
    def serve_upload_file(filename: str):
        """Отдача файлов из папки uploads."""
        if not SecurityUtils.is_safe_filename(filename):
            logger.error("🚨 Недопустимое имя файла")
            return jsonify(error="Недопустимое имя файла"), 400
        
        uploads_dir = os.path.abspath(config_manager.config.upload_dir)
        file_path = os.path.join(uploads_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"🚨 Файл не найден: {filename}")
            return jsonify(error="Файл не найден"), 404
        
        return send_from_directory(uploads_dir, filename)


    @app.route("/train_model", methods=["POST"])
    @login_required
    def train_model_route():
        """Запуск обучения модели."""
        StateManager.handle_training()
        logger.info("✅ Обучение модели начато")
        return jsonify(message="Обучение начато"), 200


    @app.route("/get_state_count")
    @login_required
    def get_state_count():
        """API для получения количества состояний."""
        try:
            db = StateManager.get_db()
            states = db.get_states()
            return jsonify(count=len(states))
        except Exception as e:
            logger.error(f"🚨 Ошибка получения количества состояний: {str(e)}")
            return jsonify(error="Ошибка получения количества состояний"), 500


    @app.route("/get_states")
    @login_required  
    def get_states():
        """API для получения списка состояний."""
        try:
            db = StateManager.get_db()
            states_dict = db.get_states()
            # Преобразуем в формат, ожидаемый фронтендом
            states_list = [{"code": code, "description": desc} for code, desc in states_dict.items()]
            return jsonify(states=states_list)
        except Exception as e:
            logger.error(f"🚨 Ошибка получения списка состояний: {str(e)}")
            return jsonify(error="Ошибка получения списка состояний"), 500


    @app.route("/get_videos/<state_code>")
    @login_required
    def get_videos(state_code: str):
        """API для получения файлов (видео и изображений) для определенного состояния."""
        try:
            page = int(request.args.get('page', 1))
            per_page = 10
            
            logger.info(f"📂 Запрос файлов для состояния '{state_code}', страница {page}")
            
            # Проверяем безопасность имени состояния
            if not SecurityUtils.is_safe_filename(state_code):
                logger.error(f"🚨 Недопустимый код состояния: {state_code}")
                return jsonify(error="Недопустимый код состояния"), 400
            
            # Путь к папке с видео для этого состояния
            state_dir = os.path.join("train", "dataset", state_code)
            logger.info(f"📁 Проверяем папку: {state_dir}")
            
            if not os.path.exists(state_dir):
                logger.warning(f"⚠️ Папка не существует: {state_dir}")
                return jsonify(files=[], total_pages=0, current_page=page, total_files=0)
            
            # Получаем список медиа файлов (видео и изображения)
            all_files = []
            all_files_in_dir = os.listdir(state_dir)
            logger.info(f"📋 Всего файлов в папке: {len(all_files_in_dir)}")
            
            video_count = 0
            image_count = 0
            other_count = 0
            
            for filename in all_files_in_dir:
                extension = filename.lower().split('.')[-1] if '.' in filename else ''
                # Проверяем видео расширения
                if extension in config_manager.config.allowed_video_extensions:
                    all_files.append(filename)
                    video_count += 1
                    logger.debug(f"📹 Видео файл: {filename}")
                # Проверяем изображения расширения  
                elif extension in config_manager.config.allowed_image_extensions:
                    all_files.append(filename)
                    image_count += 1
                    logger.debug(f"🖼️ Изображение: {filename}")
                else:
                    other_count += 1
                    logger.debug(f"❓ Неподдерживаемый файл: {filename} (расширение: {extension})")
            
            logger.info(f"📊 Найдено файлов: видео={video_count}, изображений={image_count}, других={other_count}, всего медиа={len(all_files)}")
            
            # Пагинация
            total = len(all_files)
            total_pages = (total + per_page - 1) // per_page
            start = (page - 1) * per_page
            end = start + per_page
            files = all_files[start:end]
            
            return jsonify(
                files=files,
                total_pages=total_pages,
                current_page=page,
                total_files=total
            )
            
        except Exception as e:
            logger.error(f"🚨 Ошибка получения видео для состояния {state_code}: {str(e)}")
            return jsonify(error="Ошибка получения видео"), 500


    @app.route("/delete_video/<state_code>", methods=["POST"])
    @login_required
    def delete_video(state_code: str):
        """API для удаления видео."""
        try:
            data = request.get_json()
            filename = data.get('filename')
            
            if not filename or not SecurityUtils.is_safe_filename(filename):
                return jsonify(error="Недопустимое имя файла"), 400
            
            if not SecurityUtils.is_safe_filename(state_code):
                return jsonify(error="Недопустимый код состояния"), 400
            
            # Путь к файлу в датасете
            dataset_file = os.path.join("train", "dataset", state_code, filename)
            # Путь к файлу в uploads (если есть)
            upload_file = os.path.join("uploads", filename)
            
            deleted_files = []
            
            # Удаляем из датасета
            if os.path.exists(dataset_file):
                os.remove(dataset_file)
                deleted_files.append(f"dataset/{state_code}/{filename}")
            
            # Удаляем из uploads
            if os.path.exists(upload_file):
                os.remove(upload_file)
                deleted_files.append(f"uploads/{filename}")
            
            if not deleted_files:
                return jsonify(error="Файл не найден"), 404
            
            logger.info(f"✅ Удалены файлы: {', '.join(deleted_files)}")
            return jsonify(message=f"Файл {filename} успешно удален", deleted=deleted_files)
            
        except Exception as e:
            logger.error(f"🚨 Ошибка удаления видео: {str(e)}")
            return jsonify(error="Ошибка удаления файла"), 500


    @app.route("/favicon.ico")
    def favicon():
        """Обработка запроса favicon."""
        return '', 204  # Возвращаем пустой ответ с кодом "No Content"


    @app.route("/upload_video/<state_code>", methods=["POST"])
    @login_required
    def upload_video(state_code: str):
        """API для загрузки видео в датасет."""
        try:
            if not SecurityUtils.is_safe_filename(state_code):
                return jsonify(error="Недопустимый код состояния"), 400
            
            if 'video' not in request.files:
                return jsonify(error="Файл не найден"), 400
            
            file = request.files['video']
            if file.filename == '':
                return jsonify(error="Файл не выбран"), 400
            
            if not SecurityUtils.is_safe_filename(file.filename):
                return jsonify(error="Недопустимое имя файла"), 400
            
            # Создаем папку для состояния, если её нет
            state_dir = os.path.join("train", "dataset", state_code)
            os.makedirs(state_dir, exist_ok=True)
            
            # Сохраняем файл
            file_path = os.path.join(state_dir, file.filename)
            file.save(file_path)
            
            logger.info(f"✅ Видео {file.filename} загружено в датасет для состояния {state_code}")
            return jsonify(message=f"Видео {file.filename} успешно загружено")
            
        except Exception as e:
            logger.error(f"🚨 Ошибка загрузки видео: {str(e)}")
            return jsonify(error="Ошибка загрузки файла"), 500


    @app.route("/train/dataset/<state_code>/<filename>")
    @login_required
    def serve_dataset_file(state_code: str, filename: str):
        """Отдача файлов из датасета для предпросмотра."""
        try:
            if not SecurityUtils.is_safe_filename(state_code) or not SecurityUtils.is_safe_filename(filename):
                return jsonify(error="Недопустимое имя файла"), 400
            
            dataset_dir = os.path.join("train", "dataset", state_code)
            file_path = os.path.join(dataset_dir, filename)
            
            if not os.path.exists(file_path):
                return jsonify(error="Файл не найден"), 404
            
            return send_from_directory(os.path.abspath(dataset_dir), filename)
            
        except Exception as e:
            logger.error(f"🚨 Ошибка отдачи файла из датасета: {str(e)}")
            return jsonify(error="Ошибка доступа к файлу"), 500


def register_socketio_events(socketio):
    """Регистрация SocketIO событий."""
    
    logger.info("🔧 Регистрация SocketIO событий...")
    
    @socketio.on('connect')
    def on_connect():
        """Обработка подключения клиента."""
        client_id = request.sid
        logger.info(f"🔌 Подключился клиент {client_id}")
        
        # Используем один процессор по умолчанию для всех клиентов
        if "default" not in processors:
            try:
                processors["default"] = config_manager.create_video_processor()
                logger.info("✅ Создан общий процессор по умолчанию")
            except Exception as e:
                logger.error(f"🚨 Ошибка создания процессора: {str(e)}")
                socketio.emit('error', {'message': f'Ошибка создания процессора: {str(e)}'}, to=client_id)
                return
        
        # Создаем задачи для этого клиента, но используем общий процессор
        if client_id not in background_tasks:
            background_tasks[client_id] = [
                socketio.start_background_task(StreamManager.frame_process_thread, "default"),  # Используем default процессор
                socketio.start_background_task(StreamManager.data_emit_thread, "default")      # Используем default процессор
            ]
            logger.info(f"🎬 Созданы задачи для клиента {client_id}")
        
        if not hasattr(StreamManager.frame_emit_thread, 'started'):
            socketio.start_background_task(StreamManager.frame_emit_thread)
            StreamManager.frame_emit_thread.started = True
            logger.info("📡 Запущен глобальный поток отправки кадров")


    @socketio.on('disconnect')
    def on_disconnect():
        """Обработка отключения клиента."""
        client_id = request.sid
        # Не освобождаем общий процессор, только убираем задачи клиента
        background_tasks.pop(client_id, None)
        logger.info(f"🧹 Клиент {client_id} отключён")
        # Процессор "default" остается работать для других клиентов


    @socketio.on('set_video_source')
    def handle_set_video_source(data):
        """Смена источника видео."""
        logger.info(f"📨 ПОЛУЧЕНО событие set_video_source: {data}")
        
        client_id = request.sid
        src = data.get('source') if data else None
        
        logger.info(f"🔄 Запрос смены источника от клиента {client_id}: '{src}'")
        
        if not src:
            logger.error(f"🚨 Пустой источник от клиента {client_id}")
            socketio.emit('error', {'message': 'Источник не указан'}, to=client_id)
            return
            
        if not SecurityUtils.is_safe_filename(src):
            logger.error(f"🚨 Недопустимое имя источника '{src}' от клиента {client_id}")
            socketio.emit('error', {'message': f'Недопустимое имя источника: {src}'}, to=client_id)
            return
        
        # Освобождаем старый общий процессор
        old = processors.pop("default", None)
        if old:
            old.running = False
            try:
                old.release()
                logger.info("🧹 Старый общий процессор освобождён")
            except Exception as e:
                logger.error(f"🚨 Ошибка освобождения старого процессора: {str(e)}")
        
        # Очищаем очередь кадров
        with frame_queue.mutex:
            frame_queue.queue.clear()
        
        try:
            # Определяем путь к видео
            if src == 'webcam':
                video_path = 'webcam'
            elif src == 'remote_camera':
                logger.warning("⚠️ Удаленная камера пока не реализована, используем веб-камеру")
                video_path = 'webcam'
            else:
                video_path = os.path.join(config_manager.config.upload_dir, src)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Файл не найден: {video_path}")
            
            logger.info(f"📹 Создание нового процессора для пути: {video_path}")
            new_proc = config_manager.create_video_processor(video_path)
            processors["default"] = new_proc
            logger.info("✅ Новый процессор создан и добавлен")
            
            # Обновляем задачи для всех подключенных клиентов
            client_count = len(background_tasks)
            logger.info(f"🔄 Обновление задач для {client_count} подключенных клиентов")
            
            for bg_client_id in list(background_tasks.keys()):
                background_tasks[bg_client_id] = [
                    socketio.start_background_task(StreamManager.frame_process_thread, "default"),
                    socketio.start_background_task(StreamManager.data_emit_thread, "default")
                ]
                logger.debug(f"🎬 Обновлены задачи для клиента {bg_client_id}")
            
            # Уведомляем всех клиентов об изменении источника
            socketio.emit('source_changed', {'message': f'Источник изменён на: {src}'})
            logger.info(f"✅ Источник видео успешно изменён на '{src}' (путь: {video_path})")
            
        except Exception as e:
            logger.error(f"🚨 Ошибка смены источника '{src}': {str(e)}")
            import traceback
            logger.error(f"🚨 Полная трассировка: {traceback.format_exc()}")
            socketio.emit('error', {'message': f'Ошибка смены источника: {str(e)}'}, to=client_id)


    @socketio.on('request_logs')
    def handle_log_request(data):
        """Отправка логов клиенту."""
        client_id = data.get('clientId')
        proc = processors.get("default")  # Используем общий процессор
        logs = proc.logs[-100:] if proc else []
        try:
            socketio.emit('log_update', {'logs': logs[:10]}, to=client_id)
        except Exception as e:
            logger.error(f"🚨 Ошибка отправки логов: {str(e)}")


    @socketio.on('request_chart_data')
    def handle_chart_data_request(data):
        """Отправка данных для графиков."""
        client_id = data.get('clientId')
        proc = processors.get("default")  # Используем общий процессор
        counts = proc.track_labels if proc else {}
        durations = proc.state_durations if proc else {}
        
        logger.debug(f"📊 Отправка данных графиков клиенту {client_id}: counts={len(counts)}, durations={len(durations)}")
        
        try:
            socketio.emit('chart_update', {
                'state_counts': counts,
                'state_durations': durations
            }, to=client_id)
        except Exception as e:
            logger.error(f"🚨 Ошибка отправки данных графиков: {str(e)}")
