import unittest
import sqlite3
import os
from dotenv import load_dotenv
from app.db import DatabaseManager

load_dotenv()


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        """Подготовка тестовой БД."""
        self.db_path = os.getenv('DB_PATH', "test_pig_states.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        """Очистка тестовой БД."""
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_init_db(self):
        """Тест инициализации БД."""
        states = self.db.get_states()
        expected = {
            "Feeding": "Состояние: Feeding",
            "Lateral_Lying": "Состояние: Lateral_Lying",
            "Sitting": "Состояние: Sitting",
            "Standing": "Состояние: Standing",
            "Sternal_Lying": "Состояние: Sternal_Lying"
        }
        self.assertEqual(states, expected)

    def test_add_state(self):
        """Тест добавления состояния."""
        self.assertTrue(self.db.add_state("TestState", "Тестовое состояние"))
        states = self.db.get_states()
        self.assertIn("TestState", states)
        self.assertEqual(states["TestState"], "Тестовое состояние")

    def test_add_existing_state(self):
        """Тест добавления существующего состояния."""
        self.db.add_state("TestState", "Тестовое состояние")
        self.assertFalse(self.db.add_state("TestState", "Другое описание"))

    def test_update_state(self):
        """Тест обновления состояния."""
        self.db.add_state("TestState", "Тестовое состояние")
        self.assertTrue(self.db.update_state("TestState", "NewState", "Новое состояние"))
        states = self.db.get_states()
        self.assertNotIn("TestState", states)
        self.assertIn("NewState", states)
        self.assertEqual(states["NewState"], "Новое состояние")

    def test_delete_state(self):
        """Тест удаления состояния."""
        self.db.add_state("TestState", "Тестовое состояние")
        self.assertTrue(self.db.delete_state("TestState"))
        states = self.db.get_states()
        self.assertNotIn("TestState", states)


if __name__ == '__main__':
    unittest.main()