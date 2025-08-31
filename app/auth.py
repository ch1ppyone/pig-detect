"""
Модуль аутентификации для системы мониторинга свиней.
Содержит Blueprint для входа/выхода и регистрации пользователей.
"""

from typing import Optional
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

from .db import DatabaseManager, User as DBUser
from .logging import logger


class AuthError(Exception):
    """Исключение для ошибок аутентификации."""
    pass


class User(UserMixin):
    """Класс пользователя для Flask-Login."""
    
    def __init__(self, db_user: DBUser):
        """
        Инициализация пользователя.
        
        Args:
            db_user: Объект пользователя из базы данных
        """
        self.id = str(db_user.id)  # Flask-Login требует строку
        self.username = db_user.username
        self.is_admin = db_user.is_admin
        self._db_user = db_user
    
    def __repr__(self):
        return f"<User {self.username} (admin: {self.is_admin})>"


class AuthService:
    """Сервис аутентификации."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Аутентификация пользователя."""
        try:
            with DatabaseManager(self.db_path) as db:
                db_user = db.verify_password(username, password)
                if db_user:
                    return User(db_user)
                return None
        except Exception as e:
            logger.error(f"🚨 Ошибка аутентификации пользователя {username}: {str(e)}")
            raise AuthError(f"Ошибка аутентификации: {str(e)}") from e
    
    def load_user_by_id(self, user_id: str) -> Optional[User]:
        """Загрузка пользователя по ID."""
        try:
            with DatabaseManager(self.db_path) as db:
                # Получаем пользователя напрямую из БД
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, username, password_hash, is_admin FROM users WHERE id = ?", 
                        (int(user_id),)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        db_user = DBUser(
                            id=row[0],
                            username=row[1],
                            password_hash=row[2],
                            is_admin=bool(row[3])
                        )
                        return User(db_user)
                    return None
        except Exception as e:
            logger.error(f"🚨 Ошибка загрузки пользователя {user_id}: {str(e)}")
            return None
    
    def register_user(self, username: str, password: str, is_admin: bool = False) -> bool:
        """Регистрация нового пользователя."""
        try:
            with DatabaseManager(self.db_path) as db:
                return db.add_user(username, password, is_admin)
        except Exception as e:
            logger.error(f"🚨 Ошибка регистрации пользователя {username}: {str(e)}")
            raise AuthError(f"Ошибка регистрации: {str(e)}") from e


# Глобальные переменные
_auth_service: Optional[AuthService] = None
auth_bp = Blueprint('auth', __name__)

# Инициализация менеджера входа
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Пожалуйста, войдите в систему для доступа к этой странице.'
login_manager.login_message_category = 'info'

# Функция загрузки пользователя будет зарегистрирована в init_auth

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Страница входа в систему."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Валидация входных данных
        if not username or not password:
            flash('Пожалуйста, заполните все поля', 'error')
            return render_template('login.html')

        try:
            user = _auth_service.authenticate_user(username, password)
            if user:
                login_user(user)
                logger.info(f"✅ Пользователь {username} успешно вошел в систему")
                
                next_page = request.args.get('next')
                if next_page and next_page.startswith('/'):  # Защита от redirect атак
                    return redirect(next_page)
                return redirect(url_for('index'))
            else:
                flash('Неверное имя пользователя или пароль', 'error')
                logger.warning(f"⚠️ Неудачная попытка входа: {username}")
                
        except AuthError as e:
            flash('Ошибка системы аутентификации', 'error')
            logger.error(f"🚨 Ошибка аутентификации: {str(e)}")

    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Выход из системы."""
    logger.info(f"👋 Пользователь {current_user.username} вышел из системы")
    logout_user()
    flash('Вы успешно вышли из системы', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    """Регистрация нового пользователя (только для администраторов)."""
    if not current_user.is_admin:
        flash('Доступ запрещен. Требуются права администратора.', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        is_admin = request.form.get('is_admin') == 'on'

        # Валидация данных
        if not username or not password:
            flash('Пожалуйста, заполните все обязательные поля', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('Имя пользователя должно содержать минимум 3 символа', 'error')
            return render_template('register.html')

        if len(password) < 4:
            flash('Пароль должен содержать минимум 4 символа', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Пароли не совпадают', 'error')
            return render_template('register.html')

        try:
            success = _auth_service.register_user(username, password, is_admin)
            if success:
                flash(f'Пользователь {username} успешно создан', 'success')
                logger.info(f"✅ Администратор {current_user.username} создал пользователя {username} (admin: {is_admin})")
            else:
                flash('Пользователь с таким именем уже существует', 'error')
                
        except AuthError as e:
            flash('Ошибка при создании пользователя', 'error')
            logger.error(f"🚨 Ошибка регистрации: {str(e)}")

    return render_template('register.html')

def init_auth(app, db_path: str) -> None:
    """
    Инициализация системы аутентификации.
    
    Args:
        app: Экземпляр Flask приложения
        db_path: Путь к базе данных
    """
    global _auth_service
    
    try:
        # Инициализируем сервис аутентификации
        _auth_service = AuthService(db_path)
        
        # Настраиваем Flask-Login
        login_manager.init_app(app)

        @login_manager.user_loader
        def load_user(user_id: str) -> Optional[User]:
            """Загрузка пользователя по ID для Flask-Login."""
            if not user_id:
                return None
            return _auth_service.load_user_by_id(user_id)

        # Регистрируем blueprint
        app.register_blueprint(auth_bp)
        
        logger.info(f"✅ Система аутентификации инициализирована (БД: {db_path})")
        
    except Exception as e:
        logger.error(f"🚨 Ошибка инициализации аутентификации: {str(e)}")
        raise
