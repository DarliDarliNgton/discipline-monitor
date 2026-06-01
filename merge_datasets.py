"""
Объединяет несколько датасетов в один для обучения YOLO.
Итог: datasets/combined/  с единым data.yaml

Маппинг классов:
  0 = phone_usage    (из human_phone_glasses: класс "phone" = индекс 3)
  1 = sleeping       (из sleeping датасета: указать ниже)

Bottle и food детектируются через COCO pre-training yolo11 — отдельно не обучаем.
"""

import shutil
import yaml
from pathlib import Path

# ─── Настройки ────────────────────────────────────────────────────────────────

PHONE_DIR   = Path("datasets/human_phone_glasses")
SLEEP_DIR   = Path("datasets/sleeping")   # <- заменить на реальный путь после скачивания
OUT_DIR     = Path("datasets/combined")

# Индекс класса "phone" в human_phone_glasses
# Классы: ['KacaMata'=0, 'face'=1, 'person'=2, 'phone'=3]
PHONE_CLASS_IDX = 3

# Индекс класса "sleeping" в датасете sleeping (уточним после скачивания)
SLEEP_CLASS_IDX = 0

# Итоговые классы в combined датасете
FINAL_CLASSES = ["phone_usage", "sleeping"]

# ──────────────────────────────────────────────────────────────────────────────

def remap_labels(src_label_dir: Path, dst_label_dir: Path,
                 keep_class: int, new_class: int) -> int:
    """
    Копирует label-файлы, оставляя только нужный класс и переименовывая его.
    Возвращает количество скопированных файлов.
    """
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lf in src_label_dir.glob("*.txt"):
        lines_out = []
        for line in lf.read_text().strip().split("\n"):
            parts = line.split()
            if not parts:
                continue
            if int(parts[0]) == keep_class:
                parts[0] = str(new_class)
                lines_out.append(" ".join(parts))
        if lines_out:
            (dst_label_dir / lf.name).write_text("\n".join(lines_out))
            count += 1
    return count


def copy_images(src_img_dir: Path, dst_img_dir: Path,
                label_dir: Path) -> int:
    """Копирует только те изображения, для которых есть label-файл."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_img_dir.glob("*.jpg"):
        if (label_dir / (img.stem + ".txt")).exists():
            shutil.copy2(img, dst_img_dir / img.name)
            count += 1
    for img in src_img_dir.glob("*.png"):
        if (label_dir / (img.stem + ".txt")).exists():
            shutil.copy2(img, dst_img_dir / img.name)
            count += 1
    return count


def merge(split: str) -> None:
    print(f"\n  Сплит: {split}")

    # Phone
    phone_src_lbl = PHONE_DIR / split / "labels"
    phone_src_img = PHONE_DIR / split / "images"
    dst_lbl = OUT_DIR / split / "labels"
    dst_img = OUT_DIR / split / "images"

    if phone_src_lbl.exists():
        n_lbl = remap_labels(phone_src_lbl, dst_lbl, PHONE_CLASS_IDX, 0)
        n_img = copy_images(phone_src_img, dst_img, dst_lbl)
        print(f"    phone  : {n_lbl} labels, {n_img} images")
    else:
        print(f"    phone  : папка {phone_src_lbl} не найдена, пропускаем")

    # Sleeping
    sleep_src_lbl = SLEEP_DIR / split / "labels"
    sleep_src_img = SLEEP_DIR / split / "images"

    if sleep_src_lbl.exists():
        n_lbl = remap_labels(sleep_src_lbl, dst_lbl, SLEEP_CLASS_IDX, 1)
        n_img = copy_images(sleep_src_img, dst_img, dst_lbl)
        print(f"    sleep  : {n_lbl} labels, {n_img} images")
    else:
        print(f"    sleep  : папка {sleep_src_lbl} не найдена — добавьте датасет позже")


def write_yaml() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc": len(FINAL_CLASSES),
        "names": FINAL_CLASSES,
    }
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"\n  data.yaml сохранён: {yaml_path}")
    print(f"  Классы: {FINAL_CLASSES}")


def main() -> None:
    print("=" * 50)
    print("  Слияние датасетов")
    print("=" * 50)

    if OUT_DIR.exists():
        print(f"\nУдаляю старый combined датасет: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    for split in ["train", "valid", "test"]:
        merge(split)

    write_yaml()

    # Итоговая статистика
    print("\nИтог:")
    for split in ["train", "valid", "test"]:
        img_dir = OUT_DIR / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*")))
            print(f"  {split}: {n} изображений")

    print("\nГотово! Запускайте обучение:")
    print("  python train_model.py")


if __name__ == "__main__":
    main()
