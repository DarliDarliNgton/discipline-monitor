"""
Модуль формирования отчётов о нарушениях.
Генерирует текстовый и CSV отчёты.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import List

from modules.logger import get_logger
from modules.violation_tracker import Violation

log = get_logger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "output" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_RU = {
    "sleeping": "Сон на занятии",
    "phone_usage": "Использование телефона",
    "bottle": "Запрещённый предмет (бутылка)",
    "food": "Запрещённый предмет (еда)",
}


def generate_text_report(violations: List[Violation], session_start: float) -> str:
    """Формирует текстовый отчёт и возвращает его как строку."""
    try:
        import datetime
        lines = []
        sep = "═" * 72
        thin = "─" * 72

        date_str = datetime.datetime.fromtimestamp(session_start).strftime("%Y-%m-%d")
        start_str = datetime.datetime.fromtimestamp(session_start).strftime("%H:%M:%S")
        end_str = datetime.datetime.now().strftime("%H:%M:%S")

        lines.append(sep)
        lines.append("             ОТЧЁТ О НАРУШЕНИЯХ ДИСЦИПЛИНЫ")
        lines.append(f"             Дата: {date_str}")
        lines.append(f"             Время мониторинга: {start_str} - {end_str}")
        lines.append(sep)
        lines.append("")

        if not violations:
            lines.append("  Нарушений не зафиксировано.")
        else:
            for i, v in enumerate(violations, 1):
                lines.append(f"№{i}. НАРУШЕНИЕ")
                lines.append(thin)
                start_fmt = datetime.datetime.fromtimestamp(v.start_time).strftime("%H:%M:%S")
                lines.append(f"  Время:        {start_fmt} (длительность: {v.duration:.0f} сек)")
                lines.append(f"  Тип:          {_RU.get(v.cls_name, v.cls_name)}")
                lines.append(f"  Нарушитель:   {v.person_name}")
                lines.append(f"  Уверенность:  {v.confidence:.0%}")
                if v.video_segment_path:
                    lines.append(f"  Видеозапись:  {v.video_segment_path}")
                if v.face_image_path:
                    lines.append(f"  Фото лица:    {v.face_image_path}")
                lines.append("")

        lines.append(sep)
        lines.append("                         ИТОГО")
        lines.append(sep)
        lines.append(f"  Всего нарушений:    {len(violations)}")

        known = sum(1 for v in violations if v.person_name != "Неизвестный")
        lines.append(f"  Идентифицировано:   {known}")
        lines.append(f"  Не идентифицировано:{len(violations) - known}")

        by_type: dict[str, int] = {}
        for v in violations:
            by_type[v.cls_name] = by_type.get(v.cls_name, 0) + 1
        if by_type:
            lines.append("")
            lines.append("  По типам:")
            for cls, cnt in by_type.items():
                lines.append(f"    - {_RU.get(cls, cls)}: {cnt}")
        lines.append(sep)

        return "\n".join(lines)
    except Exception as e:
        log.exception("Ошибка генерации текстового отчёта: %s", e)
        return f"Ошибка генерации отчёта: {e}"


def generate_csv_report(violations: List[Violation]) -> str:
    """Возвращает CSV-отчёт как строку (для скачивания через Streamlit)."""
    try:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["id", "type", "type_ru", "start", "duration_sec",
                        "person", "confidence", "video", "face"],
        )
        writer.writeheader()
        for v in violations:
            writer.writerow(v.to_dict())
        return output.getvalue()
    except Exception as e:
        log.exception("Ошибка генерации CSV: %s", e)
        return ""


def save_reports(violations: List[Violation], session_start: float) -> tuple[Path, Path]:
    """Сохраняет оба отчёта на диск. Возвращает (txt_path, csv_path)."""
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        txt_path = REPORTS_DIR / f"report_{ts}.txt"
        csv_path = REPORTS_DIR / f"report_{ts}.csv"

        txt_path.write_text(
            generate_text_report(violations, session_start),
            encoding="utf-8",
        )
        csv_path.write_text(
            generate_csv_report(violations),
            encoding="utf-8",
        )
        log.info("Отчёты сохранены: %s, %s", txt_path, csv_path)
        return txt_path, csv_path
    except Exception as e:
        log.exception("Ошибка сохранения отчётов: %s", e)
        raise
