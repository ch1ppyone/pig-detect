"""
Модуль для работы с базой данных SQLite.
Содержит менеджер базы данных с улучшенной обработкой ошибок и кэшированием.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from werkzeug.security import generate_password_hash, check_password_hash
from .logging import logger


@dataclass
class User:
    """Модель пользователя."""
    id: int
    username: str
    password_hash: str
    is_admin: bool


@dataclass
class State:
    """Модель состояния."""
    code: str
    description: str


class DatabaseError(Exception):
    """Базовое исключение для ошибок базы данных."""
    pass


class DatabaseManager:
    def __init__(self, db_path: str, yolo_model=None, timeout: int = 30):
        """
        Инициализация менеджера базы данных.

        Args:
            db_path: Путь к SQLite DB.
            yolo_model: Модель YOLO (опционально).
            timeout: Таймаут подключения к БД в секундах.
        """
        self.db_path = db_path
        self.timeout = timeout
        self.yolo_model = yolo_model
        self._lock = threading.RLock()  # Используем RLock для рекурсивных вызовов
        self._state_cache: Optional[Dict[str, str]] = None
        self._connection_pool = threading.local()
        
        logger.info(f"📂 Инициализация базы данных: {db_path}")
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для работы с соединением БД."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.execute("PRAGMA foreign_keys = ON")  # Включаем внешние ключи
            conn.execute("PRAGMA journal_mode = WAL")  # Используем WAL режим
            yield conn
        except sqlite3.Error as e:
            logger.error(f"🚨 Ошибка подключения к БД: {str(e)}")
            raise DatabaseError(f"Ошибка подключения к БД: {str(e)}") from e
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Создание и инициализация таблиц БД."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Создание таблиц
                    self._create_tables(cursor)
                    
                    # Инициализация данных
                    self._init_default_data(cursor)
                    
                    conn.commit()
                    logger.info("✅ База данных успешно инициализирована")
                    
            except Exception as e:
                logger.error(f"🚨 Ошибка инициализации БД: {str(e)}")
                raise DatabaseError(f"Ошибка инициализации БД: {str(e)}") from e
    
    def _create_tables(self, cursor: sqlite3.Cursor) -> None:
        """Создание таблиц в БД."""
        tables = {
            'states': """
                CREATE TABLE IF NOT EXISTS states (
                    code TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'state_translations': """
                CREATE TABLE IF NOT EXISTS state_translations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_code TEXT NOT NULL,
                    language TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(state_code, language),
                    FOREIGN KEY (state_code) REFERENCES states(code) ON DELETE CASCADE
                )
            """
        }
        
        for table_name, query in tables.items():
            cursor.execute(query)
            logger.debug(f"📋 Создана таблица: {table_name}")
    
    def _init_default_data(self, cursor: sqlite3.Cursor) -> None:
        """Инициализация данных по умолчанию."""
        # Инициализация состояний
        cursor.execute("SELECT COUNT(*) FROM states")
        states_count = cursor.fetchone()[0]
        logger.info(f"🔍 Найдено {states_count} записей в таблице states")
        
        if states_count == 0:
            default_states = [
                ("Feeding", "Состояние: Feeding"),
                ("Lateral_Lying", "Состояние: Lateral_Lying"),
                ("Sitting", "Состояние: Sitting"),
                ("Standing", "Состояние: Standing"),
                ("Sternal_Lying", "Состояние: Sternal_Lying")
            ]
            cursor.executemany("INSERT INTO states (code, description) VALUES (?, ?)", default_states)
            logger.info(f"✅ Добавлено {len(default_states)} состояний по умолчанию")
        
        # Инициализация администратора
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("admin",))
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            admin_password_hash = generate_password_hash("admin")
            cursor.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                ("admin", admin_password_hash, True)
            )
            logger.info("✅ Создан администратор по умолчанию (admin/admin)")
        
        # Инициализация переводов
        cursor.execute("SELECT COUNT(*) FROM state_translations")
        translations_count = cursor.fetchone()[0]
        
        if translations_count == 0:
            default_translations = [
                ("Feeding", "ru", "Кормление"),
                ("Sitting", "ru", "Сидит"),
                ("Standing", "ru", "Стоит"),
                ("Lateral_Lying", "ru", "Лежит на боку"),
                ("Sternal_Lying", "ru", "Лежит на животе"),
                ("Feeding", "en", "Feeding"),
                ("Sitting", "en", "Sitting"),
                ("Standing", "en", "Standing"),
                ("Lateral_Lying", "en", "Lateral Lying"),
                ("Sternal_Lying", "en", "Sternal Lying")
            ]
            cursor.executemany(
                "INSERT INTO state_translations (state_code, language, translation) VALUES (?, ?, ?)",
                default_translations
            )
            logger.info(f"✅ Добавлено {len(default_translations)} переводов по умолчанию")

    def get_states(self) -> Dict[str, str]:
        """Получение всех состояний из базы данных с кэшированием."""
        if self._state_cache is None:
            with self._lock:
                try:
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT code, description FROM states ORDER BY code")
                        self._state_cache = {row[0]: row[1] for row in cursor.fetchall()}
                        logger.info(f"🏷️ Загружено {len(self._state_cache)} состояний из БД")
                except Exception as e:
                    logger.error(f"🚨 Ошибка получения состояний: {str(e)}")
                    raise DatabaseError(f"Ошибка получения состояний: {str(e)}") from e
        return self._state_cache

    def load_class_names(self) -> List[str]:
        """Загрузка имён классов из базы данных."""
        states = self.get_states()
        class_names = list(states.keys())
        
        if not class_names:
            logger.warning("⚠️ Нет состояний в базе, используются дефолтные классы")
            class_names = ['Feeding', 'Lateral_Lying', 'Sitting', 'Standing', 'Sternal_Lying']
        
        logger.info(f"🏷️ Загружено {len(class_names)} классов: {class_names}")
        return class_names

    def add_state(self, code: str, description: str) -> bool:
        """Добавление нового состояния."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO states (code, description) VALUES (?, ?)", 
                        (code, description)
                    )
                    conn.commit()
                    
                    # Создаем папку для датасета
                    dataset_dir = os.path.join("train", "dataset", code)
                    os.makedirs(dataset_dir, exist_ok=True)
                    
                    # Сбрасываем кэш
                    self._state_cache = None
                    
                    logger.info(f"✅ Состояние добавлено: {code} - {description}")
                    return True
                    
            except sqlite3.IntegrityError:
                logger.error(f"🚨 Состояние {code} уже существует")
                return False
            except Exception as e:
                logger.error(f"🚨 Ошибка добавления состояния {code}: {str(e)}")
                raise DatabaseError(f"Ошибка добавления состояния: {str(e)}") from e

    def update_state(self, old_code: str, new_code: str, description: str) -> bool:
        """Обновление состояния."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE states SET code=?, description=? WHERE code=?", 
                        (new_code, description, old_code)
                    )
                    
                    if cursor.rowcount == 0:
                        logger.warning(f"⚠️ Состояние {old_code} не найдено для обновления")
                        return False
                    
                    conn.commit()
                    
                    # Переименовываем папку датасета если код изменился
                    if old_code != new_code:
                        old_dir = os.path.join("train", "dataset", old_code)
                        new_dir = os.path.join("train", "dataset", new_code)
                        if os.path.exists(old_dir):
                            os.rename(old_dir, new_dir)
                    
                    # Сбрасываем кэш
                    self._state_cache = None
                    
                    logger.info(f"✅ Состояние обновлено: {old_code} -> {new_code}")
                    return True
                    
            except sqlite3.IntegrityError:
                logger.error(f"🚨 Конфликт при обновлении состояния {old_code}: код {new_code} уже существует")
                return False
            except Exception as e:
                logger.error(f"🚨 Ошибка обновления состояния {old_code}: {str(e)}")
                raise DatabaseError(f"Ошибка обновления состояния: {str(e)}") from e

    def delete_state(self, code: str) -> bool:
        """Удаление состояния."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM states WHERE code=?", (code,))
                    
                    if cursor.rowcount == 0:
                        logger.warning(f"⚠️ Состояние {code} не найдено для удаления")
                        return False
                    
                    conn.commit()
                    
                    # Удаляем папку датасета
                    dataset_dir = os.path.join("train", "dataset", code)
                    if os.path.exists(dataset_dir):
                        import shutil
                        shutil.rmtree(dataset_dir)
                        logger.info(f"🗑️ Удалена папка датасета: {dataset_dir}")
                    
                    # Сбрасываем кэш
                    self._state_cache = None
                    
                    logger.info(f"✅ Состояние удалено: {code}")
                    return True
                    
            except Exception as e:
                logger.error(f"🚨 Ошибка удаления состояния {code}: {str(e)}")
                raise DatabaseError(f"Ошибка удаления состояния: {str(e)}") from e

    def get_user(self, username: str) -> Optional[User]:
        """Получение пользователя по имени."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, username, password_hash, is_admin FROM users WHERE username = ?", 
                    (username,)
                )
                row = cursor.fetchone()
                
                if row:
                    return User(
                        id=row[0],
                        username=row[1],
                        password_hash=row[2],
                        is_admin=bool(row[3])
                    )
                return None
                
        except Exception as e:
            logger.error(f"🚨 Ошибка получения пользователя {username}: {str(e)}")
            raise DatabaseError(f"Ошибка получения пользователя: {str(e)}") from e

    def verify_password(self, username: str, password: str) -> Optional[User]:
        """Проверка пароля пользователя."""
        user = self.get_user(username)
        if user and check_password_hash(user.password_hash, password):
            logger.info(f"✅ Успешная аутентификация пользователя: {username}")
            return user
        
        logger.warning(f"⚠️ Неудачная попытка входа для пользователя: {username}")
        return None

    def add_user(self, username: str, password: str, is_admin: bool = False) -> bool:
        """Добавление нового пользователя."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    password_hash = generate_password_hash(password)
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                        (username, password_hash, is_admin)
                    )
                    conn.commit()
                    
                    logger.info(f"✅ Пользователь {username} добавлен (admin: {is_admin})")
                    return True
                    
            except sqlite3.IntegrityError:
                logger.error(f"🚨 Пользователь {username} уже существует")
                return False
            except Exception as e:
                logger.error(f"🚨 Ошибка добавления пользователя {username}: {str(e)}")
                raise DatabaseError(f"Ошибка добавления пользователя: {str(e)}") from e

    def get_translations(self, language: str = 'ru') -> Dict[str, str]:
        """Получение переводов состояний для заданного языка."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state_code, translation FROM state_translations WHERE language = ? ORDER BY state_code", 
                    (language,)
                )
                translations = {row[0]: row[1] for row in cursor.fetchall()}
                logger.info(f"📖 Загружено {len(translations)} переводов для языка '{language}'")
                return translations
        except Exception as e:
            logger.error(f"🚨 Ошибка получения переводов для языка {language}: {str(e)}")
            return {}

    def get_all_translations(self) -> Dict[str, Dict[str, str]]:
        """Получение всех переводов сгруппированных по состояниям."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state_code, language, translation FROM state_translations ORDER BY state_code, language"
                )
                
                result = {}
                for state_code, language, translation in cursor.fetchall():
                    if state_code not in result:
                        result[state_code] = {}
                    result[state_code][language] = translation
                
                logger.info(f"📚 Загружены переводы для {len(result)} состояний")
                return result
        except Exception as e:
            logger.error(f"🚨 Ошибка получения всех переводов: {str(e)}")
            return {}

    def update_translation(self, state_code: str, language: str, translation: str) -> bool:
        """Обновление перевода состояния."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO state_translations (state_code, language, translation) 
                        VALUES (?, ?, ?)
                    """, (state_code, language, translation))
                    conn.commit()
                    
                    logger.info(f"✅ Обновлен перевод: {state_code} ({language}) = '{translation}'")
                    return True
                    
            except Exception as e:
                logger.error(f"🚨 Ошибка обновления перевода {state_code}: {str(e)}")
                return False

    def delete_translation(self, state_code: str, language: str) -> bool:
        """Удаление перевода состояния."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM state_translations WHERE state_code = ? AND language = ?", 
                        (state_code, language)
                    )
                    
                    if cursor.rowcount == 0:
                        logger.warning(f"⚠️ Перевод {state_code} ({language}) не найден для удаления")
                        return False
                    
                    conn.commit()
                    logger.info(f"✅ Удален перевод: {state_code} ({language})")
                    return True
                    
            except Exception as e:
                logger.error(f"🚨 Ошибка удаления перевода {state_code}: {str(e)}")
                return False
    
    def clear_cache(self) -> None:
        """Очистка кэша."""
        with self._lock:
            self._state_cache = None
            logger.debug("🧹 Кэш состояний очищен")
    
    def close(self) -> None:
        """Закрытие менеджера базы данных."""
        with self._lock:
            self._state_cache = None
            logger.info(f"🧹 Менеджер базы данных {self.db_path} закрыт")
    
    def __enter__(self):
        """Поддержка контекстного менеджера."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Автоматическое закрытие при выходе из контекста."""
        self.close()