import os
import base64
import threading
import subprocess
import shlex
import numpy as np
import cv2
from flask import render_template, jsonify, request, send_from_directory
from . import app, socketio, frame_queue, processors, background_tasks
from .config import MODEL_PATH, device, TRACKER_CONFIG, PT_MODEL_PATH, create_video_processor, is_safe_filename, TEST_VIDEO_PATH
from .logging import logger

def frame_process_thread(client_id):
    """Поток обработки кадров."""
    proc = processors.get(client_id)
    if not proc:
        socketio.emit('error', {'message': 'Процессор не найден'}, to=client_id)
        return
    try:
        while proc.running:
            frame = proc.next_frame()
            if frame is None:
                logger.warning("⚠️ Получен пустой кадр")
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
            frame_queue.put((client_id, data))
            socketio.sleep(proc.delay)
    except Exception as e:
        logger.error(f"🚨 Ошибка в потоке обработки кадров: {str(e)}")
        socketio.emit('error', {'message': f'Ошибка обработки кадра: {str(e)}'}, to=client_id)

def frame_emit_thread():
    """Поток отправки кадров."""
    while True:
        if not frame_queue.empty():
            client_id, data = frame_queue.get()
            try:
                socketio.emit('video_frame', {'data': data}, to=client_id)
            except Exception as e:
                logger.error(f"🚨 Ошибка отправки кадра: {str(e)}")
        socketio.sleep(0.05)

def data_emit_thread(client_id):
    """Поток отправки логов."""
    proc = processors.get(client_id)
    if not proc:
        socketio.emit('error', {'message': 'Процессор не найден'}, to=client_id)
        return
    try:
        while proc.running:
            logs = proc.logs[-100:]
            socketio.emit('log_update', {'logs': logs[:10]}, to=client_id)
            socketio.sleep(2)
    except Exception as e:
        logger.error(f"🚨 Ошибка отправки логов: {str(e)}")
        socketio.emit('error', {'message': f'Ошибка отправки логов: {str(e)}'}, to=client_id)

@app.route("/")
def index():
    """Главная страница."""
    files = os.listdir("test_data")
    vids = [f for f in files if f.lower().endswith(".mp4")]
    imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    sources = [(f, "Видео") for f in vids] + [(f, "Изображение") for f in imgs]

    default = processors.get("default")
    if not default:
        try:
            default = create_video_processor()
            processors["default"] = default
        except Exception as e:
            logger.error(f"🚨 Не удалось создать процессор: {str(e)}")
            return jsonify(error=f"Не удалось создать процессор: {str(e)}"), 500

    current_source = os.path.basename(default.video_path)
    return render_template("index.html", sources=sources, current_source=current_source)

@app.route("/get_states")
def get_states():
    """Получение списка состояний."""
    default = processors.get("default")
    if not default:
        try:
            default = create_video_processor()
            processors["default"] = default
        except Exception as e:
            logger.error(f"🚨 Не удалось создать процессор: {str(e)}")
            return jsonify(error=f"Не удалось создать процессор: {str(e)}"), 500
    st = default.db.get_states()
    return jsonify(states=[{"code": k, "description": v} for k, v in st.items()])

@app.route("/get_videos/<state_code>")
def get_videos(state_code):
    """Получение списка видео/изображений для состояния."""
    if not is_safe_filename(state_code):
        logger.error("🚨 Недопустимое имя состояния")
        return jsonify(error="Недопустимое имя состояния"), 400
    page = int(request.args.get('page', 1))
    per_page = 10
    fld = os.path.join("train", "dataset", state_code)
    if not os.path.isdir(fld):
        return jsonify(files=[], total_pages=0, current_page=page)

    files = [f for f in os.listdir(fld) if f.lower().endswith((".mp4", ".jpg", ".jpeg", ".png", ".bmp"))]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(fld, f)), reverse=True)

    total = len(files)
    total_pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    paginated = files[start:start + per_page]

    return jsonify(files=paginated, total_pages=total_pages, current_page=page)

@app.route("/train/dataset/<state_code>/<filename>")
def serve_dataset_file(state_code, filename):
    """Отправка файла из датасета."""
    if not is_safe_filename(state_code) or not is_safe_filename(filename):
        logger.error("🚨 Недопустимое имя файла или состояния")
        return jsonify(error="Недопустимое имя файла или состояния"), 400
    base_dir = os.path.abspath(os.path.dirname(__file__))
    directory = os.path.join(base_dir, "..", "train", "dataset", state_code)
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        logger.error("🚨 Файл не найден")
        return jsonify(error="Файл не найден"), 404
    return send_from_directory(directory, filename)

@app.route("/upload_video/<state_code>", methods=["POST"])
def upload_video(state_code):
    """Загрузка видео/изображения."""
    if not is_safe_filename(state_code):
        logger.error("🚨 Недопустимое имя состояния")
        return jsonify(error="Недопустимое имя состояния"), 400
    if "video" not in request.files:
        logger.error("🚨 Нет файла")
        return jsonify(error="Нет файла"), 400
    f = request.files["video"]
    if not f.filename.lower().endswith((".mp4", ".jpg", ".jpeg", ".png", ".bmp")):
        logger.error("🚨 Неподдерживаемый формат файла")
        return jsonify(error="Поддерживаются только MP4, JPG, JPEG, PNG, BMP"), 400
    if not is_safe_filename(f.filename):
        logger.error("🚨 Недопустимое имя файла")
        return jsonify(error="Недопустимое имя файла"), 400
    fld = os.path.join("train", "dataset", state_code)
    os.makedirs(fld, exist_ok=True)
    path = os.path.join(fld, f.filename)
    if os.path.exists(path):
        logger.error("🚨 Файл уже существует")
        return jsonify(error="Файл уже существует"), 400
    f.save(path)
    logger.info("✅ Файл успешно загружен")
    return jsonify(message="Файл загружен"), 200

@app.route("/delete_video/<state_code>", methods=["POST"])
def delete_video(state_code):
    """Удаление файла из датасета."""
    if not is_safe_filename(state_code):
        logger.error("🚨 Недопустимое имя состояния")
        return jsonify(error="Недопустимое имя состояния"), 400
    data = request.get_json()
    fn = data.get("filename")
    if not is_safe_filename(fn):
        logger.error("🚨 Недопустимое имя файла")
        return jsonify(error="Недопустимое имя файла"), 400
    path = os.path.join("train", "dataset", state_code, fn)
    if os.path.exists(path):
        os.remove(path)
        logger.info("✅ Файл успешно удалён")
        return jsonify(message="Файл удалён"), 200
    logger.error("🚨 Файл не найден")
    return jsonify(error="Файл не найден"), 400

@app.route("/train_model", methods=["POST"])
def train_model_route():
    """Запуск обучения модели."""
    def runner():
        proc = processors.get("default")
        if not proc:
            try:
                proc = create_video_processor()
                processors["default"] = proc
            except Exception as e:
                socketio.emit('training_warning', {'message': f'Ошибка инициализации процессора: {str(e)}'})
                return
        states = proc.db.get_states()
        missing = [
            s for s in states
            if not os.path.isdir(os.path.join("train", "dataset", s))
               or not os.listdir(os.path.join("train", "dataset", s))
        ]
        if missing:
            socketio.emit('training_warning', {'message': f'Нет данных для классов: {",".join(missing)}'})
            return
        cmd = "python train/train_pt.py"
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            socketio.emit('training_log', {'log': line.strip()})
        code = p.wait()
        socketio.emit('training_complete', {'message': 'OK' if code == 0 else 'ERR'})
        proc.release()
        processors["default"] = create_video_processor(proc.video_path)

    threading.Thread(target=runner, daemon=True).start()
    logger.info("✅ Обучение модели начато")
    return jsonify(message="Обучение начато"), 200

@socketio.on('connect')
def on_connect():
    """Обработка подключения клиента."""
    client_id = request.sid
    if client_id not in processors:
        try:
            processors[client_id] = create_video_processor(TEST_VIDEO_PATH)
            logger.info(f"✅ Процессор создан для клиента {client_id}")
        except Exception as e:
            logger.error(f"🚨 Ошибка создания процессора: {str(e)}")
            socketio.emit('error', {'message': f'Ошибка создания процессора: {str(e)}'}, to=client_id)
            return
    if client_id not in background_tasks:
        background_tasks[client_id] = [
            socketio.start_background_task(frame_process_thread, client_id),
            socketio.start_background_task(data_emit_thread, client_id)
        ]
    if not hasattr(frame_emit_thread, 'started'):
        socketio.start_background_task(frame_emit_thread)
        frame_emit_thread.started = True

@socketio.on('disconnect')
def on_disconnect():
    """Обработка отключения клиента."""
    client_id = request.sid
    proc = processors.pop(client_id, None)
    if proc:
        proc.running = False
        try:
            proc.release()
            logger.info(f"🧹 Процессор клиента {client_id} освобождён")
        except Exception as e:
            logger.error(f"🚨 Ошибка освобождения процессора: {str(e)}")
    background_tasks.pop(client_id, None)

@socketio.on('set_video_source')
def handle_set_video_source(data):
    """Смена источника видео."""
    client_id = request.sid
    src = data.get('source')
    if not src or not is_safe_filename(src):
        logger.error("🚨 Недопустимое имя файла")
        socketio.emit('error', {'message': 'Недопустимое имя файла'}, to=client_id)
        return
    old = processors.pop(client_id, None)
    if old:
        old.running = False
        try:
            old.release()
            logger.info(f"🧹 Старый процессор клиента {client_id} освобождён")
        except Exception as e:
            logger.error(f"🚨 Ошибка освобождения старого процессора: {str(e)}")
    with frame_queue.mutex:
        frame_queue.queue.clear()
    try:
        new_proc = create_video_processor(os.path.join("test_data", src))
        processors[client_id] = new_proc
        background_tasks[client_id] = [
            socketio.start_background_task(frame_process_thread, client_id),
            socketio.start_background_task(data_emit_thread, client_id)
        ]
        socketio.emit('source_changed', {'message': 'Источник изменён'}, to=client_id)
        logger.info(f"✅ Источник видео изменён для клиента {client_id}")
    except Exception as e:
        logger.error(f"🚨 Ошибка смены источника: {str(e)}")
        socketio.emit('error', {'message': f'Ошибка смены источника: {str(e)}'}, to=client_id)

@socketio.on('request_logs')
def handle_log_request(data):
    """Отправка логов клиенту."""
    client_id = data.get('clientId')
    proc = processors.get(client_id)
    logs = proc.logs[-100:] if proc else []
    try:
        socketio.emit('log_update', {'logs': logs[:10]}, to=client_id)
    except Exception as e:
        logger.error(f"🚨 Ошибка отправки логов: {str(e)}")

@socketio.on('request_chart_data')
def handle_chart_data_request(data):
    """Отправка данных для графиков."""
    client_id = data.get('clientId')
    proc = processors.get(client_id)
    counts = proc.track_labels if proc else {}
    durations = proc.state_durations if proc else {}
    try:
        socketio.emit('chart_update', {
            'state_counts': counts,
            'state_durations': durations
        }, to=client_id)
    except Exception as e:
        logger.error(f"🚨 Ошибка отправки данных графиков: {str(e)}")

@socketio.on('add_state')
def handle_add_state(data):
    """Добавление состояния."""
    client_id = request.sid
    code = data.get('code')
    desc = data.get('description')
    default = processors.get("default")
    if not default:
        try:
            default = create_video_processor()
            processors["default"] = default
        except Exception as e:
            logger.error(f"🚨 Ошибка создания процессора: {str(e)}")
            socketio.emit('error', {'message': f'Ошибка создания процессора: {str(e)}'}, to=client_id)
            return
    if default.db.add_state(code, desc):
        socketio.emit('state_added', {'message': 'OK'}, to=client_id)
        logger.info(f"✅ Состояние {code} добавлено")
    else:
        socketio.emit('error', {'message': 'Состояние уже существует'}, to=client_id)
        logger.error(f"🚨 Состояние {code} уже существует")

@socketio.on('update_state')
def handle_update_state(data):
    """Обновление состояния."""
    client_id = request.sid
    old = data.get('old_code')
    new = data.get('new_code')
    desc = data.get('description')
    default = processors.get("default")
    if not default:
        try:
            default = create_video_processor()
            processors["default"] = default
        except Exception as e:
            logger.error(f"🚨 Ошибка создания процессора: {str(e)}")
            socketio.emit('error', {'message': f'Ошибка создания процессора: {str(e)}'}, to=client_id)
            return
    if default.db.update_state(old, new, desc):
        socketio.emit('state_updated', {'message': 'OK'}, to=client_id)
        logger.info(f"✅ Состояние обновлено: {old} -> {new}")
    else:
        socketio.emit('error', {'message': 'Не удалось обновить состояние'}, to=client_id)
        logger.error(f"🚨 Не удалось обновить состояние {old}")

@socketio.on('delete_state')
def handle_delete_state(data):
    """Удаление состояния."""
    client_id = request.sid
    code = data.get('code')
    default = processors.get("default")
    if not default:
        try:
            default = create_video_processor()
            processors["default"] = default
        except Exception as e:
            logger.error(f"🚨 Ошибка создания процессора: {str(e)}")
            socketio.emit('error', {'message': f'Ошибка создания процессора: {str(e)}'}, to=client_id)
            return
    if default.db.delete_state(code):
        socketio.emit('state_deleted', {'message': 'OK'}, to=client_id)
        logger.info(f"✅ Состояние {code} удалено")
    else:
        socketio.emit('error', {'message': 'Не удалось удалить состояние'}, to=client_id)
        logger.error(f"🚨 Не удалось удалить состояние {code}")