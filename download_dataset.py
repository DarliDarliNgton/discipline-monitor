from roboflow import Roboflow
from pathlib import Path

API_KEY = "EmEIniR2I1SwrNPXnflW"
WORKSPACE = "test-dzcze"
PROJECT = "student-behavior-wgokl"
VERSION = 1
FORMAT = "yolov8"

DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

print(f"Скачиваю датасет: {WORKSPACE}/{PROJECT} v{VERSION}")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download(FORMAT, location=str(DATASETS_DIR / PROJECT))
print(f"\nДатасет скачан в: {DATASETS_DIR / PROJECT}")

# Показываем содержимое data.yaml
import yaml
yaml_path = DATASETS_DIR / PROJECT / "data.yaml"
if yaml_path.exists():
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    print("\ndata.yaml:")
    print(f"  Классы: {data.get('names')}")
    print(f"  nc: {data.get('nc')}")
    print(f"  train: {data.get('train')}")
    print(f"  val: {data.get('val')}")
