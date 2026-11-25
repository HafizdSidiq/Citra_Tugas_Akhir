"""Komponen tampilan bersama untuk semua halaman."""

from __future__ import annotations

import base64
import html
import re
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import streamlit as st

from core.pipeline import PipelineParams

BASE_DIR = Path(__file__).parent
THEME_CSS = BASE_DIR / "theme.css"
MOTION_CSS = BASE_DIR / "motion.css"


def bootstrap(page_title: str, page_icon: str = "🚗") -> None:
    """Inisialisasi konfigurasi halaman, gaya, dan penanda intro."""
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    css = THEME_CSS.read_text(encoding="utf-8")
    motion = MOTION_CSS.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(f"<style>{motion}</style>", unsafe_allow_html=True)
    if "intro_done" not in st.session_state:
        st.session_state["intro_done"] = False


def mark_intro_done() -> None:
    """Tandai animasi introduksi selesai setelah render pertama."""
    st.session_state["intro_done"] = True


def intro_class() -> str:
    """Kelas animasi hanya untuk render pertama."""
    return "intro-only" if not st.session_state.get("intro_done", True) else ""


def page_header(title: str, subtitle: str, eyebrow: str = "Pengolahan Citra Digital · Tugas Akhir") -> None:
    st.markdown(
        f'<div class="{intro_class()}"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def plate_svg(text: str = "B1234ABC") -> str:
    """Ilustrasi plat buatan sendiri, bukan salinan dari referensi."""
    return f"""
    <svg class="plate-illustration" width="150" height="52" viewBox="0 0 150 52"
         role="img" aria-label="Ilustrasi plat nomor">
      <rect x="2" y="2" width="146" height="48" rx="12" fill="#FFFFFF" stroke="#54485E" stroke-width="4"/>
      <circle cx="17" cy="26" r="5" fill="#DDF3EA" stroke="#54485E" stroke-width="2"/>
      <text x="79" y="33" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif"
            font-size="15" font-weight="800" letter-spacing="1.5" fill="#1A1A1A">
        {html.escape(text)}
      </text>
      <rect x="128" y="17" width="12" height="18" rx="3" fill="#F9CF5F" stroke="#54485E" stroke-width="2"/>
    </svg>
    """


def render_spotlight(title: str = "Siap membaca plat", subtitle: str = "") -> None:
    st.markdown(
        f'<div class="spotlight-card">{plate_svg()}'
        f'<div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div></div>',
        unsafe_allow_html=True,
    )


def render_badge(label: str, positive: bool = True) -> None:
    cls = "badge badge-up" if positive else "badge badge-down"
    st.markdown(f'<span class="{cls}">{html.escape(label)}</span>', unsafe_allow_html=True)


def _split_stat_value(value: str) -> tuple[int, str]:
    """Pisahkan angka dari satuan untuk animasi counter CSS."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)(.*)$", str(value))
    if not match:
        return 0, str(value)
    return int(float(match.group(1))), match.group(2)


def render_stat_grid(stats: Iterable[tuple[str, str, str, bool]]) -> None:
    """Render 3 kartu statistik: (label, nilai, keterangan, naik?)."""
    cards = []
    for label, value, note, positive in stats:
        badge_cls = "badge badge-up" if positive else "badge badge-down"
        target, suffix = _split_stat_value(value)
        cards.append(
            f'<div class="stat-card"><div class="label-small">{html.escape(label)}</div>'
            f'<div class="stat-value" style="--target:{target}">'
            f'<span class="counter-number" data-value="{target}"></span>'
            f'<span class="counter-suffix">{html.escape(suffix)}</span></div>'
            f'<span class="{badge_cls}">{html.escape(note)}</span></div>'
        )
    st.markdown(f'<div class="stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_performance_panel(
    metrics: Iterable[tuple[str, float, str]], note: str = ""
) -> None:
    """Panel kanan kuning dengan bilah progres. Nilai 0-100."""
    rows = []
    for label, value, display in metrics:
        pct = max(0.0, min(100.0, float(value)))
        rows.append(
            '<div class="perf-metric">'
            f'<div class="perf-metric-head"><span>{html.escape(label)}</span>'
            f'<span>{html.escape(str(display))}</span></div>'
            '<div class="progress-track">'
            f'<div class="progress-fill" style="width:{pct:.1f}%"></div>'
            "</div></div>"
        )
    note_html = f'<div class="perf-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        '<div class="perf-zone"><h2 class="perf-title">Ringkasan performa</h2>'
        '<p class="perf-subtitle">Diperbarui dari hasil terakhir</p>'
        f'{"".join(rows)}{note_html}</div>',
        unsafe_allow_html=True,
    )


def render_image_card(
    image_bgr: Any, caption: str, css_class: str = "stage-card", delay_ms: int = 0
) -> None:
    """Tampilkan citra sebagai HTML agar animasi CSS bisa diterapkan."""
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        st.warning("Citra tahap tidak dapat ditampilkan.")
        return
    data = base64.b64encode(buffer.tobytes()).decode("ascii")
    delay = f' style="animation-delay:{int(delay_ms)}ms"' if css_class == "stage-card" else ""
    st.markdown(
        f'<div class="{css_class}"{delay}><img alt="{html.escape(caption)}" '
        f'src="data:image/png;base64,{data}">'
        f'<div class="stage-caption">{html.escape(caption)}</div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        f'<div class="empty-state"><strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(message)}</span></div>',
        unsafe_allow_html=True,
    )


def render_scan_overlay() -> None:
    """Garis pindai yang menyapu saat tombol Baca plat ditekan."""
    # SVG untuk garis kotak disuntik lewat st.iframe karena sanitizer markdown
    # Streamlit membuang SVG pada markup jenis ini.
    css = THEME_CSS.read_text(encoding="utf-8") + MOTION_CSS.read_text(encoding="utf-8")
    st.iframe(
        f'<style>{css}</style>'
        '<div class="scan-frame" style="height:92px">'
        '<svg class="plate-outline" viewBox="0 0 600 180" preserveAspectRatio="none">'
        '<rect x="12" y="12" width="576" height="156" rx="18"></rect></svg>'
        '<div class="scan-line"></div></div>',
        width="stretch",
        height=108,
    )


def render_result_text(text: str) -> None:
    """Tampilkan teks plat dengan satu karakter per langkah steps()."""
    if not text:
        return
    spans = []
    step = min(160, int(1120 / max(1, len(text) - 1)))
    for i, char in enumerate(text):
        spans.append(
            f'<span style="animation-delay:{300 + i * step}ms">{html.escape(char)}</span>'
        )
    st.markdown(
        f'<div class="result-text plate-text">{"".join(spans)}</div>',
        unsafe_allow_html=True,
    )


def render_failure_card(message: str) -> None:
    st.markdown(
        f'<div class="failure-card" style="background:var(--rose);border-radius:var(--radius-card);'
        f'padding:16px;font-size:13px;line-height:1.5">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_history(items: list[dict[str, Any]]) -> None:
    if not items:
        render_empty_state("Belum ada yang dibaca", "Unggah foto di halaman Deteksi untuk mulai.")
        return
    rows = []
    for item in items[:6]:
        text = html.escape(str(item.get("text", "?")))
        conf = float(item.get("confidence", 0.0))
        when = html.escape(str(item.get("time", "")))
        rows.append(
            f'<div class="history-item"><span class="plate-text" style="font-size:16px">{text}</span>'
            f'<span class="history-time">{conf:.0f}% · {when}</span></div>'
        )
    st.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)


def parameter_widgets() -> PipelineParams:
    """Widget parameter penting; slider Streamlit commit ketika dilepas."""
    with st.expander("Parameter pengolahan", expanded=False):
        left, right = st.columns(2)
        with left:
            low = st.slider("Ambang Canny bawah", 20, 120, 50, 5, key="canny_low")
            high = st.slider("Ambang Canny atas", 60, 240, 150, 10, key="canny_high")
            min_area = st.slider(
                "Luas minimum (× citra)",
                0.0002,
                0.005,
                0.001,
                0.0001,
                format="%.4f",
                key="min_area",
            )
        with right:
            aspect_min = st.slider("Rasio minimum", 1.4, 3.0, 1.6, 0.1, key="aspect_min")
            aspect_max = st.slider("Rasio maksimum", 2.8, 4.2, 3.8, 0.1, key="aspect_max")
            use_clahe = st.checkbox("Gunakan CLAHE", value=True, key="use_clahe")
            binarization = st.selectbox(
                "Binisasi",
                ["auto", "otsu", "adaptive"],
                index=0,
                key="binarization",
                help="auto mencoba keduanya dan memilih keyakinan OCR tertinggi.",
            )
    high = max(high, low + 10)
    aspect_max = max(aspect_max, aspect_min + 0.2)
    return PipelineParams(
        canny_low=low,
        canny_high=high,
        min_area_ratio=float(min_area),
        aspect_min=float(aspect_min),
        aspect_max=float(aspect_max),
        use_clahe=use_clahe,
        binarization=binarization,
    )


def append_history(text: str, confidence: float) -> None:
    """Simpan ringkasan bacaan terakhir di session state."""
    history = list(st.session_state.get("history", []))
    history.insert(
        0,
        {
            "text": text or "?",
            "confidence": round(float(confidence), 1),
            "time": time.strftime("%H:%M"),
        },
    )
    st.session_state["history"] = history[:12]


def performance_from_result(result: Any) -> list[tuple[str, float, str]]:
    """Ubah PipelineResult menjadi tiga metrik ringkas untuk panel kanan."""
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    total_ms = float(getattr(result, "timings_ms", {}).get("total", 0.0) or 0.0)
    n_plates = len(getattr(result, "plates", []) or [])
    speed = max(0.0, 100.0 - total_ms / 10.0)
    return [
        ("Keyakinan OCR", confidence, f"{confidence:.0f}%"),
        ("Plat terlokalisasi", min(100.0, n_plates * 50), f"{n_plates} plat"),
        ("Kecepatan", speed, f"{total_ms:.0f} ms"),
    ]
