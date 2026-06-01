"""
Модуль детекции нарушений через YOLO.
Загружает модель один раз, предоставляет метод detect(frame).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from modules.logger import get_logger

log = get_logger(__name__)

# Маппинг COCO классов → наши типы нарушений
# YOLO pretrained на COCO уже детектирует эти объекты
COCO_TO_INTERNAL = {
    39: "bottle",                                        # bottle
    41: "bottle",                                        # cup
    46: "food", 47: "food", 48: "food", 49: "food",     # banana, apple, sandwich, orange
    51: "food", 52: "food", 53: "food",                  # pizza, donut, cake
    67: "phone_usage",                                   # cell phone
    # sleeping — нет в COCO, требует кастомной модели
}

CLASS_LABELS_RU = {
    "sleeping":    "Sleep",
    "phone_usage": "Phone",
    "bottle":      "Bottle",
    "food":        "Food",
}

CLASS_COLORS = {
    "sleeping":    (0, 0, 255),
    "phone_usage": (0, 165, 255),
    "bottle":      (0, 255, 255),
    "food":        (255, 0, 255),
}

Detection = Tuple[str, float, Tuple[int, int, int, int]]


class ViolationDetector:
    def __init__(self, model_path: str | Path, confidence: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._model = None

        log.info("Инициализация детектора. Модель: %s", self.model_path)
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            if not self.model_path.exists():
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
            self._model = YOLO(str(self.model_path))
            log.info("Модель успешно загружена: %s", self.model_path)
        except FileNotFoundError as e:
            log.error("Модель не найдена: %s", e)
            raise
        except Exception as e:
            log.exception("Ошибка при загрузке модели: %s", e)
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Запускает детекцию на кадре.
        Возвращает список (class_name, confidence, (x1,y1,x2,y2)).
        """
        if self._model is None:
            log.error("Модель не загружена, детекция невозможна")
            return []
        try:
            results = self._model.predict(
                frame,
                conf=self.confidence,
                verbose=False,
            )
            detections: List[Detection] = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls_id in COCO_TO_INTERNAL:
                        cls_name = COCO_TO_INTERNAL[cls_id]
                    else:
                        continue
                    detections.append((cls_name, conf, (x1, y1, x2, y2)))
            log.debug("Детекций найдено: %d", len(detections))
            return detections
        except Exception as e:
            log.exception("Ошибка детекции на кадре: %s", e)
            return []

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        enabled_classes: set[str] | None = None,
        active_timers: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Рисует боксы, подписи и таймеры нарушений на копии кадра."""
        try:
            import time
            annotated = frame.copy()
            for cls_name, conf, (x1, y1, x2, y2) in detections:
                if enabled_classes and cls_name not in enabled_classes:
                    continue
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_ru = CLASS_LABELS_RU.get(cls_name, cls_name)
                # Добавляем таймер если нарушение активно
                if active_timers and cls_name in active_timers:
                    elapsed = int(time.time() - active_timers[cls_name])
                    label = f"{label_ru}  {elapsed}s"
                else:
                    label = f"{label_ru} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )
            return annotated
        except Exception as e:
            log.exception("Ошибка аннотирования кадра: %s", e)
            return frame

    def set_confidence(self, confidence: float) -> None:
        self.confidence = max(0.1, min(1.0, confidence))
        log.debug("Порог уверенности изменён на %.2f", self.confidence)
