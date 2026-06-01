"""
Скрипт обучения YOLO-модели на датасете нарушений дисциплины.
Запуск: python train_model.py

Перед запуском:
1. Скачайте датасет с Roboflow и распакуйте в папку datasets/
2. Укажите путь к data.yaml ниже
3. Запустите скрипт (лучше в Google Colab с GPU)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# ─── Настройки ───────────────────────────────────────────────────────────────
DATA_YAML = "datasets/combined/data.yaml"  # объединённый датасет
BASE_MODEL = "yolo11s.pt"                  # nano=быстро, small=баланс
EPOCHS = 30
IMG_SIZE = 640
BATCH = 16
PROJECT_NAME = "violations_detector"
# ─────────────────────────────────────────────────────────────────────────────


def train() -> None:
    print("=" * 60)
    print("  Обучение модели детекции нарушений дисциплины")
    print("=" * 60)

    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"\n[ОШИБКА] Файл датасета не найден: {data_path}")
        print("Скачайте датасет с Roboflow Universe и распакуйте в datasets/")
        print("Рекомендуемые датасеты:")
        print("  - https://universe.roboflow.com/  (поиск: sleeping, phone detection)")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ОШИБКА] ultralytics не установлен. Выполните: pip install ultralytics")
        sys.exit(1)

    try:
        print(f"\nЗагружаем базовую модель: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)

        print(f"Начинаем обучение...")
        print(f"  Датасет:  {DATA_YAML}")
        print(f"  Эпохи:    {EPOCHS}")
        print(f"  Размер:   {IMG_SIZE}")
        print(f"  Батч:     {BATCH}")
        print()

        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            name=PROJECT_NAME,
            patience=20,
            save=True,
            plots=True,
        )

        # Копируем лучшую модель в models/best.pt
        best_src = Path(f"runs/detect/{PROJECT_NAME}/weights/best.pt")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        dest = models_dir / "best.pt"

        if best_src.exists():
            import shutil
            shutil.copy(best_src, dest)
            print(f"\n✅ Модель сохранена в: {dest}")
        else:
            print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Файл {best_src} не найден, скопируйте вручную")

        print("\n" + "=" * 60)
        print("  Обучение завершено!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    train()
