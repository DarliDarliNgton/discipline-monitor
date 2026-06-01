"""
Модуль трекинга нарушений.
Фиксирует нарушение только если оно длится дольше порога.
После исчезновения продолжает запись ещё buffer_after секунд.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from modules.logger import get_logger

log = get_logger(__name__)


@dataclass
class Violation:
    violation_id: int
    cls_name: str
    start_time: float
    end_time: Optional[float] = None
    confidence: float = 0.0
    face_image_path: Optional[str] = None
    video_segment_path: Optional[str] = None
    person_name: str = "Неизвестный"

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    def to_dict(self) -> dict:
        import datetime
        return {
            "id": self.violation_id,
            "type": self.cls_name,
            "type_ru": _RU.get(self.cls_name, self.cls_name),
            "start": datetime.datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S"),
            "duration_sec": round(self.duration),
            "person": self.person_name,
            "confidence": f"{self.confidence:.0%}",
            "video": self.video_segment_path or "—",
            "face": self.face_image_path or "—",
        }


_RU = {
    "sleeping": "Сон",
    "phone_usage": "Телефон",
    "bottle": "Бутылка",
    "food": "Еда",
}

# Минимальное время непрерывного детектирования для фиксации нарушения (сек)
_MIN_PERSIST = 1.0


@dataclass
class _ActiveCandidate:
    cls_name: str
    first_seen: float
    last_seen: float
    max_conf: float
    confirmed: bool = False


class ViolationTracker:
    def __init__(self, buffer_after: float = 5.0) -> None:
        self.buffer_after = buffer_after
        self._candidates: Dict[str, _ActiveCandidate] = {}
        self._active_violations: Dict[str, Violation] = {}
        self._finished: List[Violation] = []
        self._id_counter = 0
        log.info("ViolationTracker инициализирован. Буфер после: %.1f с", buffer_after)

    def update(
        self,
        detected_classes: Set[str],
        confidences: Dict[str, float],
        enabled_classes: Set[str] | None = None,
    ) -> List[Violation]:
        """
        Вызывается на каждом обработанном кадре.
        detected_classes — множество имён классов на текущем кадре.
        Возвращает список только что ЗАВЕРШЁННЫХ нарушений.
        """
        try:
            now = time.time()
            if enabled_classes:
                detected_classes = detected_classes & enabled_classes

            newly_finished: List[Violation] = []

            # Обновляем/создаём кандидатов
            for cls in detected_classes:
                if cls not in self._candidates:
                    self._candidates[cls] = _ActiveCandidate(
                        cls_name=cls,
                        first_seen=now,
                        last_seen=now,
                        max_conf=confidences.get(cls, 0.0),
                    )
                    log.debug("Новый кандидат: %s", cls)
                else:
                    c = self._candidates[cls]
                    c.last_seen = now
                    c.max_conf = max(c.max_conf, confidences.get(cls, 0.0))

                # Подтверждаем кандидата как нарушение
                c = self._candidates[cls]
                if not c.confirmed and (now - c.first_seen) >= _MIN_PERSIST:
                    c.confirmed = True
                    self._id_counter += 1
                    v = Violation(
                        violation_id=self._id_counter,
                        cls_name=cls,
                        start_time=c.first_seen,
                        confidence=c.max_conf,
                    )
                    self._active_violations[cls] = v
                    log.info("Нарушение зафиксировано: %s (id=%d)", cls, v.violation_id)

            # Проверяем исчезновение
            to_remove_candidates = []
            for cls, c in self._candidates.items():
                if cls not in detected_classes:
                    elapsed_since_last = now - c.last_seen
                    if elapsed_since_last > self.buffer_after:
                        to_remove_candidates.append(cls)
                        if cls in self._active_violations:
                            v = self._active_violations.pop(cls)
                            v.end_time = c.last_seen + self.buffer_after
                            self._finished.append(v)
                            newly_finished.append(v)
                            log.info(
                                "Нарушение завершено: %s (id=%d, длительность=%.1f с)",
                                cls, v.violation_id, v.duration,
                            )

            for cls in to_remove_candidates:
                del self._candidates[cls]

            return newly_finished
        except Exception as e:
            log.exception("Ошибка в ViolationTracker.update: %s", e)
            return []

    @property
    def active_violations(self) -> List[Violation]:
        return list(self._active_violations.values())

    @property
    def all_violations(self) -> List[Violation]:
        return self._finished + list(self._active_violations.values())

    def reset(self) -> None:
        self._candidates.clear()
        self._active_violations.clear()
        self._finished.clear()
        self._id_counter = 0
        log.info("Трекер сброшен")

    def set_buffer(self, seconds: float) -> None:
        self.buffer_after = max(1.0, float(seconds))
        log.debug("Буфер после установлен: %.1f с", self.buffer_after)
