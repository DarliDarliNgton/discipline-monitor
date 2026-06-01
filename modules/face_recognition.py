"""
Модуль распознавания лиц (постобработка).
Использует InsightFace для извлечения эмбеддингов.
База данных хранится в data/faces/ — по одной папке на студента.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.logger import get_logger

log = get_logger(__name__)

FACES_DIR = Path(__file__).parent.parent / "data" / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = FACES_DIR / "embeddings.pkl"
OUTPUT_FACES_DIR = Path(__file__).parent.parent / "output" / "faces"
OUTPUT_FACES_DIR.mkdir(parents=True, exist_ok=True)


class FaceRecognizer:
    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold
        self._app = None
        self._db: Dict[str, np.ndarray] = {}
        self._available = False

        self._load_insightface()
        if self._available:
            self._load_db()

    def _load_insightface(self) -> None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(320, 320))
            self._available = True
            log.info("InsightFace успешно инициализирован")
        except ImportError:
            log.warning("insightface не установлен — распознавание лиц отключено")
        except Exception as e:
            log.exception("Ошибка инициализации InsightFace: %s", e)

    def _load_db(self) -> None:
        if DB_FILE.exists():
            try:
                with open(DB_FILE, "rb") as f:
                    self._db = pickle.load(f)
                log.info("База лиц загружена: %d студентов", len(self._db))
            except Exception as e:
                log.exception("Ошибка загрузки базы лиц: %s", e)
                self._db = {}
        else:
            log.info("База лиц пуста — файл %s не найден", DB_FILE)

    def _save_db(self) -> None:
        try:
            with open(DB_FILE, "wb") as f:
                pickle.dump(self._db, f)
            log.info("База лиц сохранена: %d студентов", len(self._db))
        except Exception as e:
            log.exception("Ошибка сохранения базы лиц: %s", e)

    def register_face(self, name: str, image: np.ndarray) -> bool:
        """Добавляет лицо студента в базу данных."""
        if not self._available:
            log.warning("InsightFace недоступен — регистрация невозможна")
            return False
        try:
            faces = self._app.get(image)
            if not faces:
                log.warning("Лицо не обнаружено на изображении для '%s'", name)
                return False
            emb = faces[0].normed_embedding
            self._db[name] = emb
            self._save_db()
            log.info("Студент '%s' зарегистрирован в базе", name)
            return True
        except Exception as e:
            log.exception("Ошибка регистрации лица '%s': %s", name, e)
            return False

    def identify(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Определяет личность по изображению.
        Возвращает (имя, схожесть). Имя = 'Неизвестный' если не распознан.
        """
        if not self._available or not self._db:
            return "Неизвестный", 0.0
        try:
            faces = self._app.get(image)
            if not faces:
                log.debug("Лицо не обнаружено при идентификации")
                return "Неизвестный", 0.0

            emb = faces[0].normed_embedding
            best_name, best_sim = "Неизвестный", 0.0
            for name, db_emb in self._db.items():
                sim = float(np.dot(emb, db_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_sim >= self.similarity_threshold:
                log.debug("Идентифицирован: %s (sim=%.3f)", best_name, best_sim)
                return best_name, best_sim
            return "Неизвестный", best_sim
        except Exception as e:
            log.exception("Ошибка идентификации лица: %s", e)
            return "Неизвестный", 0.0

    def process_segment(self, video_path: str, violation_id: int) -> Tuple[str, Optional[str]]:
        """
        Постобработка сегмента: находит лучший кадр с лицом,
        идентифицирует, сохраняет изображение лица.
        Возвращает (имя_нарушителя, путь_к_фото_лица).
        """
        if not self._available:
            return "Неизвестный", None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log.error("Не удалось открыть видео для постобработки: %s", video_path)
                return "Неизвестный", None

            best_face_img = None
            best_face_size = 0
            frame_idx = 0
            step = 10

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    faces = self._app.get(frame)
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        size = (x2 - x1) * (y2 - y1)
                        if size > best_face_size:
                            best_face_size = size
                            best_face_img = frame[max(0,y1):y2, max(0,x1):x2].copy()
                frame_idx += 1

            cap.release()

            if best_face_img is None:
                log.debug("Лицо не найдено в сегменте: %s", video_path)
                return "Неизвестный", None

            # Сохраняем фото лица
            import time
            ts = time.strftime("%Y%m%d_%H%M%S")
            face_path = OUTPUT_FACES_DIR / f"face_{ts}_{violation_id:03d}.jpg"
            cv2.imwrite(str(face_path), best_face_img)

            # Идентифицируем
            name, sim = self.identify(best_face_img)
            log.info("Постобработка сегмента завершена: %s → '%s' (sim=%.3f)", video_path, name, sim)
            return name, str(face_path)

        except Exception as e:
            log.exception("Ошибка постобработки сегмента %s: %s", video_path, e)
            return "Неизвестный", None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def student_count(self) -> int:
        return len(self._db)

    @property
    def student_names(self) -> List[str]:
        return list(self._db.keys())
