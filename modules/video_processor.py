"""
Оркестратор обработки видеопотока.
Объединяет детектор, трекер и писатель.
Используется как из Streamlit (покадрово), так и из WebRTC-процессора.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Set

import cv2
import numpy as np

from modules.detector import ViolationDetector
from modules.students_db import StudentsDB
from modules.logger import get_logger
from modules.report_generator import save_reports
from modules.video_writer import SegmentWriter
from modules.violation_tracker import Violation, ViolationTracker

log = get_logger(__name__)


class VideoProcessor:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        buffer_after: float = 5.0,
        frame_stride: int = 1,
        enabled_classes: Optional[Set[str]] = None,
        on_violation_start: Optional[Callable[[Violation], None]] = None,
        on_violation_end: Optional[Callable[[Violation], None]] = None,
    ) -> None:
        self.frame_stride = frame_stride
        self.enabled_classes = enabled_classes or {"sleeping", "phone_usage", "bottle", "food"}
        self.on_violation_start = on_violation_start
        self.on_violation_end = on_violation_end

        self._frame_counter = 0
        self._session_start = time.time()
        self._lock = threading.Lock()
        self._stopped = False

        try:
            self.detector = ViolationDetector(model_path, confidence)
            self.tracker = ViolationTracker(buffer_after)
            self.writer = SegmentWriter(fps=25.0)
            self.face_rec = StudentsDB()
            log.info("VideoProcessor инициализирован")
        except Exception as e:
            log.exception("Ошибка инициализации VideoProcessor: %s", e)
            raise

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Обрабатывает один кадр.
        Возвращает аннотированный кадр.
        """
        if self._stopped:
            return frame
        try:
            self._frame_counter += 1

            # Всегда пишем в пре-буфер
            self.writer.push_frame(frame)

            # Детектируем только каждый N-й кадр
            if self._frame_counter % self.frame_stride != 0:
                return frame

            detections = self.detector.detect(frame)

            if detections:
                log.info("Детекции: %s", [(d[0], f"{d[1]:.0%}") for d in detections])

            detected_classes = {d[0] for d in detections if d[0] in self.enabled_classes}
            confidences = {d[0]: d[1] for d in detections}

            if detected_classes:
                log.info("Классы нарушений: %s (enabled: %s)", detected_classes, self.enabled_classes)

            # Обновляем трекер
            finished = self.tracker.update(detected_classes, confidences, self.enabled_classes)

            # Реакция на завершённые нарушения
            for v in finished:
                seg_path = self.writer.stop_segment()
                if seg_path:
                    v.video_segment_path = str(seg_path)
                if self.on_violation_end:
                    threading.Thread(
                        target=self._post_process,
                        args=(v,),
                        daemon=True,
                    ).start()
                else:
                    log.debug("Нарушение завершено без постобработки: id=%d", v.violation_id)

            # Реакция на активные нарушения — начинаем запись + распознавание
            for v in self.tracker.active_violations:
                if not self.writer.is_recording:
                    try:
                        path = self.writer.start_segment(v.violation_id, v.cls_name, frame.shape)
                        v.video_segment_path = str(path)
                    except Exception as e:
                        log.error("Не удалось начать запись сегмента: %s", e)
                    if self.on_violation_start:
                        self.on_violation_start(v)
                if v.person_name == "Неизвестный" and self.face_rec.is_available:
                    if self._frame_counter % (self.frame_stride * 5) == 0:
                        name, sim = self.face_rec.identify(frame)
                        if sim > 0.0:
                            with self._lock:
                                v.person_name = name
                            log.info("Распознан: '%s' (sim=%.3f) для нарушения id=%d", name, sim, v.violation_id)

            # Таймеры для отображения на видео
            active_timers = {
                v.cls_name: v.start_time
                for v in self.tracker.active_violations
            }
            return self.detector.annotate(frame, detections, self.enabled_classes, active_timers)
        except Exception as e:
            log.exception("Ошибка обработки кадра: %s", e)
            return frame

    def _post_process(self, violation: Violation) -> None:
        """Постобработка сегмента в фоновом потоке — распознавание лица."""
        try:
            if violation.video_segment_path and self.face_rec.is_available:
                cap = cv2.VideoCapture(violation.video_segment_path)
                if cap.isOpened():
                    best_name = "Неизвестный"
                    best_sim = 0.0
                    frame_idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % 15 == 0:
                            name, sim = self.face_rec.identify(frame)
                            if sim > best_sim:
                                best_sim = sim
                                best_name = name
                        frame_idx += 1
                    cap.release()
                    with self._lock:
                        violation.person_name = best_name
                    log.info("Постобработка: %s -> '%s' (sim=%.3f)", violation.video_segment_path, best_name, best_sim)
            if self.on_violation_end:
                self.on_violation_end(violation)
        except Exception as e:
            log.exception("Ошибка постобработки нарушения id=%d: %s", violation.violation_id, e)

    def get_all_violations(self) -> List[Violation]:
        with self._lock:
            return self.tracker.all_violations

    def get_active_violations(self) -> List[Violation]:
        return self.tracker.active_violations

    def save_report(self) -> tuple:
        violations = self.get_all_violations()
        return save_reports(violations, self._session_start)

    def update_settings(
        self,
        confidence: Optional[float] = None,
        buffer_after: Optional[float] = None,
        frame_stride: Optional[int] = None,
        enabled_classes: Optional[Set[str]] = None,
    ) -> None:
        try:
            if confidence is not None:
                self.detector.set_confidence(confidence)
            if buffer_after is not None:
                self.tracker.set_buffer(buffer_after)
            if frame_stride is not None:
                self.frame_stride = max(1, int(frame_stride))
            if enabled_classes is not None:
                self.enabled_classes = enabled_classes
            log.debug("Настройки обновлены")
        except Exception as e:
            log.exception("Ошибка обновления настроек: %s", e)

    def stop(self) -> None:
        self._stopped = True
        self.writer.release()
        log.info("VideoProcessor остановлен")

    def reset(self) -> None:
        self._stopped = False
        self._frame_counter = 0
        self._session_start = time.time()
        self.tracker.reset()
        log.info("VideoProcessor сброшен")
