import os
import sqlite3
import threading
from .logging import logger


class DatabaseManager:
    def __init__(self, db_path, yolo_model=None):
        """
        Инициализация менеджера базы данных.

        Args:
            db_path (str): Путь к SQLite DB.
            yolo_model: Модель YOLO (опционально).
        """
        self.db_path = db_path
        self.state_cache = None
        self.yolo_model = yolo_model
        self.lock = threading.Lock()
        logger.info(f"📂 Инициализация базы данных: {db_path}")
        self.init_db()

    def init_db(self):
        """Создание/инициализация таблицы states."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    c = conn.cursor()
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS states (
                            code TEXT PRIMARY KEY,
                            description TEXT NOT NULL
                        )
                    """)
                    c.execute("SELECT COUNT(*) FROM states")
                    count = c.fetchone()[0]
                    logger.info(f"🔍 Найдено {count} записей в таблице states")
                    if count == 0:
                        initial = [
                            ("Feeding", "Состояние: Feeding"),
                            ("Lateral_Lying", "Состояние: Lateral_Lying"),
                            ("Sitting", "Состояние: Sitting"),
                            ("Standing", "Состояние: Standing"),
                            ("Sternal_Lying", "Состояние: Sternal_Lying")
                        ]
                        c.executemany("INSERT INTO states VALUES (?, ?)", initial)
                        conn.commit()
                        logger.info(f"✅ База данных заполнена состояниями: {initial}")
            except sqlite3.OperationalError as e:
                logger.error(f"🚨 Ошибка инициализации базы данных: {str(e)}")
                raise

    def get_states(self):
        """Получение всех состояний из базы данных."""
        if self.state_cache is None:
            with self.lock:
                try:
                    with sqlite3.connect(self.db_path, timeout=30) as conn:
                        c = conn.cursor()
                        c.execute("SELECT code, description FROM states")
                        self.state_cache = {r[0]: r[1] for r in c.fetchall()}
                        logger.info(f"🏷️ Состояния из базы данных: {self.state_cache}")
                except sqlite3.OperationalError as e:
                    logger.error(f"🚨 Ошибка получения состояний: {str(e)}")
                    raise
        return self.state_cache

    def load_class_names(self):
        """Загрузка имён классов из базы данных."""
        states = self.get_states()
        class_names = list(states.keys())
        if not class_names:
            logger.warning("⚠️ Нет состояний в базе, используются дефолтные классы")
            class_names = ['Feeding', 'Lateral_Lying', 'Sitting', 'Standing', 'Sternal_Lying']
        logger.info(f"🏷️ Имена классов из базы данных: {class_names}")
        return class_names

    def add_state(self, code, desc):
        """Добавление нового состояния."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO states VALUES (?, ?)", (code, desc))
                    conn.commit()
                    os.makedirs(os.path.join("train", "states", code), exist_ok=True)
                    self.state_cache = None
                    logger.info(f"✅ Состояние добавлено: {code}")
                    return True
            except sqlite3.IntegrityError:
                logger.error(f"🚨 Не удалось добавить состояние {code}: уже существует")
                return False
            except sqlite3.OperationalError as e:
                logger.error(f"🚨 Ошибка добавления состояния: {str(e)}")
                raise

    def update_state(self, old, new, desc):
        """Обновление состояния."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    c = conn.cursor()
                    c.execute("UPDATE states SET code=?, description=? WHERE code=?", (new, desc, old))
                    if c.rowcount:
                        conn.commit()
                        if old != new:
                            os.rename(
                                os.path.join("train", "states", old),
                                os.path.join("train", "states", new)
                            )
                        self.state_cache = None
                        logger.info(f"✅ Состояние обновлено: {old} -> {new}")
                        return True
                    return False
            except sqlite3.IntegrityError:
                logger.error(f"🚨 Не удалось обновить состояние {old}: конфликт кода")
                return False
            except sqlite3.OperationalError as e:
                logger.error(f"🚨 Ошибка обновления состояния: {str(e)}")
                raise

    def delete_state(self, code):
        """Удаление состояния."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    c = conn.cursor()
                    c.execute("DELETE FROM states WHERE code=?", (code,))
                    if c.rowcount:
                        conn.commit()
                        folder = os.path.join("train", "states", code)
                        if os.path.isdir(folder):
                            for f in os.listdir(folder):
                                os.remove(os.path.join(folder, f))
                            os.rmdir(folder)
                        self.state_cache = None
                        logger.info(f"✅ Состояние удалено: {code}")
                        return True
                    return False
            except sqlite3.OperationalError as e:
                logger.error(f"🚨 Ошибка удаления состояния: {str(e)}")
                raise

    def close(self):
        """Закрытие соединения с базой данных."""
        self.state_cache = None
        logger.info(f"🧹 Соединение с базой данных {self.db_path} закрыто")