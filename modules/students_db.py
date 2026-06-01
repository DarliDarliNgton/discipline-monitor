"""
Модуль базы данных студентов.
Регистрация через фото, хранение в data/faces/
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.logger import get_logger

log = get_logger(__name__)

FACES_DIR = Path(__file__).parent.parent / "data" / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = FACES_DIR / "embeddings.pkl"
PHOTOS_DIR = FACES_DIR / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


class StudentsDB:
    def __init__(self) -> None:
        self._db: Dict[str, dict] = {}  # name -> {embedding, photo_path}
        self._face_app = None
        self._available = False
        self._load_insightface()
        self._load_db()

    def _load_insightface(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                name="buffalo_sc", providers=["CPUExecutionProvider"]
            )
            self._face_app.prepare(ctx_id=0, det_size=(320, 320))
            self._available = True
            log.info("InsightFace готов для базы студентов")
        except Exception as e:
            log.warning("InsightFace недоступен: %s", e)

    def _load_db(self) -> None:
        if DB_FILE.exists():
            try:
                with open(DB_FILE, "rb") as f:
                    self._db = pickle.load(f)
                log.info("База студентов загружена: %d записей", len(self._db))
            except Exception as e:
                log.exception("Ошибка загрузки базы: %s", e)
                self._db = {}

    def _save_db(self) -> None:
        try:
            with open(DB_FILE, "wb") as f:
                pickle.dump(self._db, f)
        except Exception as e:
            log.exception("Ошибка сохранения базы: %s", e)

    def add_student(self, name: str, image: np.ndarray) -> Tuple[bool, str]:
        """
        Добавляет студента по фото.
        Возвращает (успех, сообщение).
        """
        if not name.strip():
            return False, "Введите имя студента"
        if not self._available:
            return False, "InsightFace не установлен (pip install insightface)"
        try:
            faces = self._face_app.get(image)
            if not faces:
                return False, "Лицо не обнаружено на фото. Попробуйте другое изображение."
            if len(faces) > 1:
                return False, f"На фото обнаружено {len(faces)} лиц. Загрузите фото с одним человеком."

            emb = faces[0].normed_embedding
            # Сохраняем фото
            photo_path = PHOTOS_DIR / f"{name.replace(' ', '_')}.jpg"
            cv2.imwrite(str(photo_path), image)

            self._db[name] = {
                "embedding": emb,
                "photo_path": str(photo_path),
            }
            self._save_db()
            log.info("Студент добавлен: %s", name)
            return True, f"Студент '{name}' успешно зарегистрирован"
        except Exception as e:
            log.exception("Ошибка добавления студента %s: %s", name, e)
            return False, f"Ошибка: {e}"

    def remove_student(self, name: str) -> bool:
        if name in self._db:
            photo = self._db[name].get("photo_path")
            if photo and Path(photo).exists():
                Path(photo).unlink(missing_ok=True)
            del self._db[name]
            self._save_db()
            log.info("Студент удалён: %s", name)
            return True
        return False

    def identify(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
        """Определяет личность. Возвращает (имя, схожесть)."""
        if not self._available or not self._db:
            return "Неизвестный", 0.0
        try:
            faces = self._face_app.get(image)
            if not faces:
                return "Неизвестный", 0.0
            emb = faces[0].normed_embedding
            best_name, best_sim = "Неизвестный", 0.0
            for name, data in self._db.items():
                sim = float(np.dot(emb, data["embedding"]))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            if best_sim >= threshold:
                return best_name, best_sim
            return "Неизвестный", best_sim
        except Exception as e:
            log.exception("Ошибка идентификации: %s", e)
            return "Неизвестный", 0.0

    def get_photo(self, name: str) -> Optional[np.ndarray]:
        if name in self._db:
            path = self._db[name].get("photo_path")
            if path and Path(path).exists():
                return cv2.imread(path)
        return None

    @property
    def students(self) -> List[str]:
        return sorted(self._db.keys())

    @property
    def count(self) -> int:
        return len(self._db)

    @property
    def is_available(self) -> bool:
        return self._available
