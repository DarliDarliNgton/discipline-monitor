"""
Система мониторинга дисциплины на занятиях.
streamlit run app.py
"""
from __future__ import annotations
import time, tempfile, traceback
from pathlib import Path
import cv2, numpy as np, streamlit as st
from modules.logger import get_logger
from modules.report_generator import generate_csv_report, generate_text_report

log = get_logger(__name__)
MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

st.set_page_config(page_title="Мониторинг дисциплины", page_icon="📹",
                   layout="wide", initial_sidebar_state="expanded")

# ─── State ────────────────────────────────────────────────────────────────────
for k, v in {"processor": None, "violations": [], "running": False,
             "session_start": None, "error_log": [], "activity_log": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _log_error(msg):
    log.error(msg); st.session_state.error_log.append(msg)

def _enabled_classes():
    return {c for c, k in [("sleeping","en_sleeping"),("phone_usage","en_phone"),
            ("bottle","en_bottle"),("food","en_food")] if st.session_state.get(k, True)}

def _get_processor():
    if st.session_state.processor is None:
        try:
            from modules.video_processor import VideoProcessor
            st.session_state.processor = VideoProcessor(
                model_path=str(MODEL_PATH),
                confidence=st.session_state.get("conf_threshold", 0.5),
                buffer_after=st.session_state.get("buffer_after", 5.0),
                frame_stride=st.session_state.get("frame_stride", 1),
                enabled_classes=_enabled_classes(),
                on_violation_end=lambda v: st.session_state.activity_log.append((time.time(), v.cls_name)))
        except Exception as e:
            _log_error(f"Ошибка: {e}\n{traceback.format_exc()}"); return None
    return st.session_state.processor

def _update():
    p = st.session_state.processor
    if p: p.update_settings(confidence=st.session_state.get("conf_threshold",0.5),
        buffer_after=st.session_state.get("buffer_after",5.0),
        frame_stride=st.session_state.get("frame_stride",1), enabled_classes=_enabled_classes())


# ─── Цикл видео ──────────────────────────────────────────────────────────────
def _run_video(cap, processor, frame_ph, journal_ph):
    n = 0
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret: break
        try:
            out = processor.process_frame(frame)
            frame_ph.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.session_state.violations = processor.get_all_violations()
            n += 1
            if n % 10 == 0:
                _update_journal(journal_ph)
        except Exception as e:
            _log_error(str(e))
    cap.release(); processor.stop(); st.session_state.running = False
    _update_journal(journal_ph)

def _update_journal(ph):
    v = st.session_state.violations
    if not v:
        ph.info("Нарушений пока нет")
        return
    import pandas as pd
    rows = [x.to_dict() for x in reversed(v)]
    df = pd.DataFrame(rows)[["id","type_ru","start","duration_sec","person","confidence"]]
    df.columns = ["#","Тип","Время","Длит.(с)","Кто","Увер."]
    ph.dataframe(df, use_container_width=True, hide_index=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📹 Мониторинг")
    page = st.radio("Раздел", ["Мониторинг", "База студентов", "История"], key="nav")

    if page == "Мониторинг":
        st.divider()
        source = st.radio("Источник", ["Веб-камера", "Видеофайл", "URL-поток"], key="src")
        st.divider()
        st.slider("Порог уверенности", 0.1, 1.0, 0.5, 0.05, key="conf_threshold", on_change=_update)
        st.slider("Буфер записи (сек)", 1, 30, 5, key="buffer_after", on_change=_update)
        st.slider("Каждый N-й кадр", 1, 5, 1, key="frame_stride", on_change=_update)
        st.divider()
        c1, c2 = st.columns(2)
        c1.checkbox("📱 Телефон", True, key="en_phone", on_change=_update)
        c2.checkbox("🍶 Бутылка", True, key="en_bottle", on_change=_update)
        c1.checkbox("🍕 Еда", True, key="en_food", on_change=_update)
        st.divider()
        if st.button("🗑️ Очистить журнал", use_container_width=True):
            st.session_state.violations = []; st.session_state.activity_log = []
            if st.session_state.processor: st.session_state.processor.tracker.reset()
        viol = st.session_state.violations
        if viol:
            st.divider()
            s = st.session_state.session_start or time.time()
            st.download_button("📄 Скачать TXT", generate_text_report(viol,s).encode("utf-8"),
                "report.txt", "text/plain", use_container_width=True)
            st.download_button("📊 Скачать CSV", generate_csv_report(viol).encode("utf-8"),
                "report.csv", "text/csv", use_container_width=True)
            try:
                from modules.excel_report import generate_excel
                st.download_button("📗 Скачать Excel", generate_excel(viol,s),
                    "report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            except: pass
    else:
        source = None


# ─── Ошибки ──────────────────────────────────────────────────────────────────
if st.session_state.error_log:
    with st.expander(f"⚠️ Ошибки ({len(st.session_state.error_log)})", expanded=False):
        for e in st.session_state.error_log[-10:]: st.error(e)
        if st.button("Очистить ошибки"): st.session_state.error_log = []; st.rerun()


# ─── Мониторинг ──────────────────────────────────────────────────────────────
if page == "Мониторинг":
    if not MODEL_PATH.exists():
        st.error(f"Модель не найдена: {MODEL_PATH}")

    # --- Заголовок + метрики ---
    st.markdown("## 📹 Система мониторинга дисциплины")
    v = st.session_state.violations
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Всего нарушений", len(v))
    m2.metric("Активных", len(st.session_state.processor.get_active_violations()) if st.session_state.processor else 0)
    m3.metric("Опознано", sum(1 for x in v if x.person_name != "Неизвестный"))
    m4.metric("Источник", source or "—")
    st.divider()

    # --- Видео + Журнал ---
    col_video, col_journal = st.columns([3, 2])

    with col_journal:
        st.markdown("#### 📋 Журнал нарушений")
        journal_ph = st.empty()
        _update_journal(journal_ph)

    with col_video:
        # === Веб-камера ===
        if source == "Веб-камера":
            st.markdown("#### 📷 Веб-камера")
            btn1, btn2 = st.columns(2)
            start_cam = btn1.button("▶  ЗАПУСТИТЬ КАМЕРУ", use_container_width=True, key="btn_cam_start")
            stop_cam = btn2.button("⏹  ОСТАНОВИТЬ", use_container_width=True, key="btn_cam_stop")
            if stop_cam: st.session_state.running = False
            frame_ph = st.empty()
            if start_cam:
                p = _get_processor()
                if p:
                    p.reset(); st.session_state.running = True; st.session_state.session_start = time.time()
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        _run_video(cap, p, frame_ph, journal_ph)
                    else:
                        st.error("❌ Камера недоступна")
                        st.session_state.running = False

        # === Видеофайл ===
        elif source == "Видеофайл":
            st.markdown("#### 📁 Видеофайл")
            uploaded = st.file_uploader("Выберите видео", type=["mp4","avi","mov","mkv"])
            btn1, btn2 = st.columns(2)
            start_f = btn1.button("▶  ЗАПУСТИТЬ", use_container_width=True, key="btn_file_start")
            stop_f = btn2.button("⏹  ОСТАНОВИТЬ", use_container_width=True, key="btn_file_stop")
            if stop_f: st.session_state.running = False
            frame_ph = st.empty()
            if start_f:
                if not uploaded:
                    st.warning("Сначала загрузите файл");
                else:
                    p = _get_processor()
                    if p:
                        p.reset(); st.session_state.running = True; st.session_state.session_start = time.time()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
                            f.write(uploaded.read())
                            cap = cv2.VideoCapture(f.name)
                        if cap.isOpened():
                            _run_video(cap, p, frame_ph, journal_ph)
                        else:
                            st.error("❌ Не удалось открыть видео")

        # === URL ===
        elif source == "URL-поток":
            st.markdown("#### 🌐 URL / RTSP поток")
            url = st.text_input("Введите адрес потока", placeholder="rtsp://... или http://...")
            btn1, btn2 = st.columns(2)
            start_u = btn1.button("▶  ПОДКЛЮЧИТЬСЯ", use_container_width=True, key="btn_url_start")
            stop_u = btn2.button("⏹  ОСТАНОВИТЬ", use_container_width=True, key="btn_url_stop")
            if stop_u: st.session_state.running = False
            frame_ph = st.empty()
            if start_u:
                if not url:
                    st.warning("Введите URL")
                else:
                    p = _get_processor()
                    if p:
                        p.reset(); st.session_state.running = True; st.session_state.session_start = time.time()
                        cap = cv2.VideoCapture(url)
                        if cap.isOpened():
                            _run_video(cap, p, frame_ph, journal_ph)
                        else:
                            st.error(f"❌ Не удалось подключиться: {url}")
                            st.session_state.running = False


# ─── База студентов ───────────────────────────────────────────────────────────
elif page == "База студентов":
    st.markdown("## 👤 База студентов")
    try:
        from modules.students_db import StudentsDB
        if "students_db" not in st.session_state:
            st.session_state.students_db = StudentsDB()
        db = st.session_state.students_db
    except Exception as e:
        st.error(str(e)); st.stop()

    if not db.is_available:
        st.warning("InsightFace не установлен: `pip install insightface onnxruntime`")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Добавить студента")
        name = st.text_input("ФИО", placeholder="Иванов Иван")
        photo = st.file_uploader("Фото (одно лицо)", type=["jpg","jpeg","png"])
        if photo: st.image(photo, width=200)
        if st.button("➕ Добавить в базу", use_container_width=True, key="add_stud"):
            if photo and name:
                img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
                ok, msg = db.add_student(name, img)
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()
            else: st.warning("Введите имя и загрузите фото")

    with col_b:
        st.markdown(f"#### В базе: {db.count} чел.")
        if db.count == 0:
            st.info("Добавьте студентов — система будет определять нарушителей по имени")
        for sname in db.students:
            with st.expander(f"👤 {sname}"):
                p = db.get_photo(sname)
                if p is not None: st.image(cv2.cvtColor(p, cv2.COLOR_BGR2RGB), width=120)
                if st.button("Удалить", key=f"d_{sname}"):
                    db.remove_student(sname); st.rerun()


# ─── История ──────────────────────────────────────────────────────────────────
elif page == "История":
    st.markdown("## 📂 История")
    t1, t2 = st.tabs(["📄 Отчёты", "🎬 Видеозаписи"])
    with t1:
        rdir = Path(__file__).parent / "output" / "reports"
        reps = sorted(rdir.glob("*.txt"), reverse=True)[:20] if rdir.exists() else []
        if not reps: st.info("Отчётов пока нет")
        for r in reps:
            with st.expander(r.stem):
                st.text(r.read_text(encoding="utf-8"))
                st.download_button("Скачать", r.read_bytes(), r.name, key=f"d_{r.stem}")
    with t2:
        sdir = Path(__file__).parent / "output" / "segments"
        segs = sorted(list(sdir.glob("*.webm")) + list(sdir.glob("*.mp4")), key=lambda x: x.stat().st_mtime, reverse=True)[:10] if sdir.exists() else []
        if not segs: st.info("Видеозаписей пока нет")
        for s in segs:
            st.caption(f"{s.name} — {s.stat().st_size//1024} KB")
            st.video(str(s))
