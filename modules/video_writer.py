"""
Модуль записи видеосегментов нарушений.
Кольцевой буфер хранит последние N секунд кадров.
При старте нарушения — буфер сбрасывается в файл, запись продолжается.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np

from modules.logger import get_logger

log = get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "segments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SegmentWriter:
    def __init__(self, fps: float = 25.0, pre_buffer_sec: float = 3.0) -> None:
        self.fps = fps
        self.pre_buffer_sec = pre_buffer_sec
        self._pre_buffer: deque = deque(maxlen=int(fps * pre_buffer_sec))
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._recording = False
        log.info(
            "SegmentWriter инициализирован. FPS=%.1f, пре-буфер=%.1f с",
            fps, pre_buffer_sec,
        )

    def push_frame(self, frame: np.ndarray) -> None:
        """Добавляет кадр в пре-буфер и/или активную запись."""
        try:
            with self._lock:
                self._pre_buffer.append(frame.copy())
                if self._recording and self._writer is not None:
                    self._writer.write(frame)
        except Exception as e:
            log.exception("Ошибка записи кадра: %s", e)

    def start_segment(self, violation_id: int, cls_name: str, frame_shape: tuple) -> Path:
        """Начинает запись нового сегмента. Возвращает путь к файлу."""
        try:
            with self._lock:
                if self._recording:
                    self._close_writer()

                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"seg_{ts}_{violation_id:03d}_{cls_name}.webm"
                path = OUTPUT_DIR / filename
                h, w = frame_shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"VP80")
                self._writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
                if not self._writer.isOpened():
                    raise RuntimeError(f"Не удалось открыть VideoWriter для {path}")

                # Сначала пишем пре-буфер
                for f in self._pre_buffer:
                    self._writer.write(f)
                self._pre_buffer.clear()

                self._current_path = path
                self._recording = True
                log.info("Запись сегмента начата: %s", path)
                return path
        except Exception as e:
            log.exception("Ошибка старта записи сегмента: %s", e)
            raise

    def stop_segment(self) -> Optional[Path]:
        """Завершает запись. Возвращает путь к файлу."""
        try:
            with self._lock:
                path = self._current_path
                self._close_writer()
                if path:
                    log.info("Запись сегмента завершена: %s", path)
                return path
        except Exception as e:
            log.exception("Ошибка остановки записи сегмента: %s", e)
            return None

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False
        self._current_path = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def release(self) -> None:
        with self._lock:
            self._close_writer()
        log.debug("SegmentWriter освобождён")
