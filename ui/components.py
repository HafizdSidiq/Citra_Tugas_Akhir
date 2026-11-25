"""Komponen tampilan bersama, mengikuti gaya docs/referensi/lukis.html."""

from __future__ import annotations

import base64
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import streamlit as st

from core.pipeline import PipelineParams

BASE_DIR = Path(__file__).parent
THEME_CSS = BASE_DIR / "theme.css"

WARNA_STAT = ("stat-putih", "stat-ungu", "stat-kuning", "stat-tosca", "stat-pink")


def bootstrap() -> None:
    """Suntik gaya. Konfigurasi halaman ada di app.py."""
    st.markdown(
        f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
    if "intro_done" not in st.session_state:
        st.session_state["intro_done"] = False


def mark_intro_done() -> None:
    """Tandai animasi introduksi selesai setelah render pertama."""
    st.session_state["intro_done"] = True


def intro_class() -> str:
    """Kelas animasi hanya untuk render pertama."""
    return "intro-only" if not st.session_state.get("intro_done", True) else ""


def page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "Pengolahan Citra Digital · Tugas Akhir",
) -> None:
    st.markdown(
        f'<div class="{intro_class()}"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_spotlight(title: str, subtitle: str) -> None:
    """Kartu ungu padat, tanpa garis dekoratif."""
    st.markdown(
        f'<div class="card"><strong style="font-size:18px">'
        f'{html.escape(title)}</strong>'
        f'<p style="margin:6px 0 0">{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_badge(label: str, positive: bool = True) -> None:
    cls = "badge-up" if positive else "badge-down"
    st.markdown(f'<span class="{cls}">{html.escape(label)}</span>', unsafe_allow_html=True)


def _split_stat_value(value: str) -> tuple[int, str]:
    """Pisahkan angka dari satuan untuk animasi penghitung."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)(.*)$", str(value))
    if not match:
        return 0, str(value)
    return int(float(match.group(1))), match.group(2)


def render_stat_grid(stats: Iterable[tuple[str, str, str, bool]]) -> None:
    """Kartu statistik bergantian warna: (label, nilai, catatan, penanda)."""
    cards = []
    for index, (label, value, note, _positive) in enumerate(stats):
        kelas = WARNA_STAT[index % len(WARNA_STAT)]
        target, suffix = _split_stat_value(value)
        cards.append(
            f'<div class="stat-card {kelas}" style="--urutan:{index}">'
            f'<div class="label-small">{html.escape(label)}</div>'
            f'<div class="stat-value" style="--target:{target}">'
            f'<span class="counter-number" data-value="{html.escape(str(target))}"></span>'
            f'<span class="counter-suffix">{html.escape(suffix)}</span></div>'
            f'<span class="badge">{html.escape(note)}</span></div>'
        )
    st.markdown(f'<div class="stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_summary_card(rows: Iterable[tuple[Any, ...]], note: str = "") -> None:
    """Ringkasan hasil nyata; tidak memakai bilah progres."""
    baris = []
    for row in rows:
        if len(row) >= 3:
            label, _nilai, tampilan = row[0], row[1], row[2]
        else:
            label, tampilan = row[0], row[1]
        baris.append(
            f'<div class="ringkas-baris"><span>{html.escape(str(label))}</span>'
            f'<strong>{html.escape(str(tampilan))}</strong></div>'
        )
    catatan = f'<p style="font-size:12px;margin:12px 0 0">{html.escape(note)}</p>' if note else ""
    st.markdown(
        f'<div class="kartu-ringkas"><strong>Ringkasan hasil</strong>'
        f'{"".join(baris)}{catatan}</div>',
        unsafe_allow_html=True,
    )


def render_image_card(
    image_bgr: Any,
    caption: str,
    css_class: str = "stage-card",
    delay_ms: int = 0,
) -> None:
    """Tampilkan citra sebagai HTML agar animasi masuk bisa diterapkan."""
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        st.warning("Citra tahap tidak dapat ditampilkan.")
        return
    data = base64.b64encode(buffer.tobytes()).decode("ascii")
    gaya = f' style="--tunda:{int(delay_ms)}ms"' if css_class == "stage-card" else ""
    st.markdown(
        f'<div class="{css_class}"{gaya}><img alt="{html.escape(caption)}" '
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
    """Garis pindai menyapu; SVG disuntik lewat iframe Streamlit."""
    css = """
      html,body{margin:0;background:transparent;overflow:hidden}
      .bingkai{position:relative;height:96px;border-radius:18px;background:#f1ecff;overflow:hidden}
      .bingkai svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
      .bingkai rect{fill:none;stroke:#ff6b9d;stroke-width:5;stroke-dasharray:1400;stroke-dashoffset:1400;
        animation:gambar 700ms ease-out 300ms 1 both}
      .garis{position:absolute;inset:0 auto 0 0;width:18%;
        background:linear-gradient(90deg,transparent,rgba(255,107,157,.6),transparent);
        animation:sapu 900ms ease-in-out 1 both}
      @keyframes sapu{0%{transform:translateX(-110%);opacity:0}12%,88%{opacity:1}100%{transform:translateX(610%);opacity:0}}
      @keyframes gambar{to{stroke-dashoffset:0}}
      @media(prefers-color-scheme:dark){.bingkai{background:#241f47}}
      @media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important}
        .bingkai rect{stroke-dashoffset:0}.garis{display:none}}
    """
    st.iframe(
        f"<style>{css}</style>"
        '<div class="bingkai">'
        '<svg viewBox="0 0 600 180" preserveAspectRatio="none" aria-hidden="true">'
        '<rect x="12" y="12" width="576" height="156" rx="18"></rect></svg>'
        '<div class="garis"></div></div>',
        width="stretch",
        height=104,
    )


def render_result_text(text: str) -> None:
    """Teks plat dengan animasi membalik per karakter."""
    if not text:
        return
    spans = "".join(
        f'<span style="animation-delay:{i * 90}ms">{html.escape(char)}</span>'
        for i, char in enumerate(text)
    )
    st.markdown(f'<div class="result-text">{spans}</div>', unsafe_allow_html=True)


def render_plate_headline(text: str, confidence: float, n_plates: int) -> None:
    """Teks plat besar, keyakinan sebagai pil berwarna."""
    if not text:
        return
    spans = "".join(
        f'<span style="animation-delay:{i * 90}ms">{html.escape(char)}</span>'
        for i, char in enumerate(text)
    )
    st.markdown(
        '<div class="result-headline">'
        f'<div class="result-text">{spans}</div>'
        f'<span class="pil-keyakinan">Keyakinan {float(confidence):.0f}%</span> '
        f'<span style="font-size:13px">{int(n_plates)} plat terdeteksi</span></div>',
        unsafe_allow_html=True,
    )


def render_copy_button(text: str, label: str = "Salin teks plat") -> None:
    """Tombol salin memakai Clipboard API di dalam iframe mandiri."""
    if not text:
        return
    payload = json.dumps(text).replace("</", "<\\/")
    css = (
        "body{margin:0;font-family:Figtree,sans-serif}"
        "button{background:#6d3cff;color:#fff;border:0;border-radius:30px;"
        "padding:10px 20px;font-weight:700;cursor:pointer;min-height:44px}"
        "button:hover{transform:translateY(-3px) rotate(-1.5deg);"
        "box-shadow:0 8px 0 -3px #ffd23f}"
        "button:active{transform:scale(.96)}"
        "button:focus-visible{outline:3px solid #ff6b9d;outline-offset:3px}"
        "span{font-size:12px;margin-left:8px;color:#5d5878}"
        "@media(prefers-color-scheme:dark){span{color:#b4aed1}}"
        "@media(prefers-reduced-motion:reduce){button:hover{transform:none;box-shadow:none}}"
    )
    st.iframe(
        f"<style>{css}</style>"
        f'<button id="salin" type="button">{html.escape(label)}</button>'
        f'<span id="pesan" aria-live="polite"></span>'
        f"<script>const teks={payload};"
        "document.getElementById('salin').addEventListener('click',async()=>{"
        "try{await navigator.clipboard.writeText(teks);}catch(e){"
        "const a=document.createElement('textarea');a.value=teks;"
        "document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();}"
        "document.getElementById('pesan').textContent='Tersalin';});</script>",
        width="stretch",
        height=52,
    )


def render_info_card(title: str, message: str) -> None:
    st.markdown(
        f'<div class="info-card"><strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(message)}</span></div>',
        unsafe_allow_html=True,
    )


def render_failure_card(message: str) -> None:
    st.markdown(
        f'<div class="failure-card"><strong>Perlu diperiksa</strong>'
        f'<span>{html.escape(message)}</span></div>',
        unsafe_allow_html=True,
    )


def render_history(items: list[dict[str, Any]]) -> None:
    if not items:
        render_empty_state("Belum ada yang dibaca", "Unggah foto di halaman Baca Plat untuk mulai.")
        return
    rows = []
    for item in items[:6]:
        teks = html.escape(str(item.get("text", "?")))
        conf = float(item.get("confidence", 0.0))
        when = html.escape(str(item.get("time", "")))
        rows.append(
            f'<div class="history-item"><span class="result-text" style="font-size:16px">{teks}</span>'
            f'<span class="history-time">{conf:.0f}% · {when}</span></div>'
        )
    st.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)


def parameter_widgets() -> PipelineParams:
    """Widget parameter penting; slider Streamlit commit ketika dilepas."""
    with st.expander("Parameter pengolahan", expanded=False):
        kiri, kanan = st.columns(2)
        with kiri:
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
        with kanan:
            aspect_min = st.slider("Rasio minimum", 1.4, 3.0, 1.6, 0.1, key="aspect_min")
            aspect_max = st.slider("Rasio maksimum", 2.8, 4.2, 3.8, 0.1, key="aspect_max")
            use_clahe = st.checkbox("Gunakan CLAHE", value=True, key="use_clahe")
            binarization = st.selectbox(
                "Binerisasi",
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


def performance_from_result(result: Any) -> list[tuple[str, str]]:
    """Ubah PipelineResult menjadi ringkasan ringkas tanpa bilah progres."""
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    total_ms = float(getattr(result, "timings_ms", {}).get("total", 0.0) or 0.0)
    n_plates = len(getattr(result, "plates", []) or [])
    return [
        ("Keyakinan OCR", f"{confidence:.0f}%"),
        ("Plat terdeteksi", f"{n_plates}"),
        ("Waktu pemrosesan", f"{total_ms:.0f} ms"),
    ]


# --------------------------------------------------------------------------
# Beranda: panggung simulasi, marquee, dan bento (CSS murni, tanpa JS kecuali
# iframe yang memang diisolasi dari markdown).
# --------------------------------------------------------------------------
TAHAP_SIMULASI = [
    ("01", "Perkecil citra", "#6d3cff"),
    ("02", "Abu-abu", "#00c4cc"),
    ("03", "CLAHE", "#ffd23f"),
    ("04", "Canny", "#ff6b9d"),
    ("05", "Kontur", "#6d3cff"),
    ("06", "Binerisasi", "#00c4cc"),
    ("07", "Segmentasi", "#ffd23f"),
    ("08", "OCR", "#ff6b9d"),
]


def render_hero_stage() -> None:
    """Panggung simulasi delapan tahap, berjalan dengan CSS saja."""
    panels = []
    for index, (nomor, nama, warna) in enumerate(TAHAP_SIMULASI):
        panels.append(
            '<section class="panel" style="--w:' + warna + '">'
            f'<span class="nomor">{nomor}</span>'
            '<div class="plat"><i></i><b>B1234ABC</b><i></i></div>'
            f'<span class="nama">{html.escape(nama)}</span></section>'
        )
    css = """
      @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@500;700&family=Unbounded:wght@600;800&display=swap');
      html,body{margin:0;background:transparent;font-family:Figtree,sans-serif}
      .panggung{position:relative;width:100%;height:420px;border-radius:30px;overflow:hidden;
        background:#6d3cff;box-shadow:0 40px 70px -34px rgba(20,17,43,.6)}
      .jalur{display:flex;width:800%;height:100%;animation:putar 16s steps(8,end) infinite}
      .panel{width:12.5%;height:100%;flex:none;display:flex;flex-direction:column;
        align-items:center;justify-content:center;gap:22px;background:#6d3cff;color:#fff;position:relative}
      .panel::before{content:"";position:absolute;width:78%;aspect-ratio:4/3;border-radius:26px;
        background:rgba(255,255,255,.08)}
      .nomor{font:800 70px Unbounded,sans-serif;opacity:.22;position:relative}
      .plat{position:relative;display:flex;align-items:center;gap:10px;padding:16px 22px;
        border-radius:20px;background:#fff;color:#14112b;box-shadow:0 18px 30px -16px rgba(0,0,0,.6)}
      .plat i{width:12px;height:38px;border-radius:6px;background:var(--w)}
      .plat b{font:800 24px Unbounded,sans-serif;letter-spacing:.06em}
      .nama{position:relative;font:700 18px Figtree,sans-serif;padding:8px 20px;border-radius:30px;
        background:var(--w);color:#14112b}
      .garis{position:absolute;left:0;right:0;height:14px;top:0;pointer-events:none;
        background:linear-gradient(90deg,transparent,#ffd23f,transparent);
        animation:sapu 2.2s ease-in-out infinite}
      .keterangan{position:absolute;left:14px;bottom:14px;z-index:2;padding:6px 14px;border-radius:30px;
        background:#14112b;color:#fff;font-size:12px;font-weight:700}
      @keyframes putar{to{transform:translateX(-87.5%)}}
      @keyframes sapu{0%{transform:translateY(0);opacity:0}15%,85%{opacity:1}100%{transform:translateY(406px);opacity:0}}
      @media(prefers-color-scheme:dark){.panggung{box-shadow:none}.plat i{background:var(--w)}}
      @media(prefers-reduced-motion:reduce){.jalur,.garis{animation:none!important}.garis{display:none}}
      @media(max-width:560px){.panggung{height:330px}.plat b{font-size:18px}.nomor{font-size:54px}}
    """
    st.iframe(
        f"<style>{css}</style>"
        '<div class="panggung"><div class="jalur">' + "".join(panels) + "</div>"
        '<div class="garis"></div>'
        '<span class="keterangan">Simulasi ilustratif · bukan hasil OCR sungguhan</span></div>',
        width="stretch",
        height=430,
    )


def render_marquee(items: Iterable[str]) -> None:
    """Marquee chip bergerak, berhenti saat reduced motion."""
    daftar = [html.escape(str(item)) for item in items]
    isi = "".join(
        f'<span class="chip-lukis" style="background:{warna}">{label}</span>'
        for label, warna in zip(
            daftar + daftar,
            ["#ffd23f", "#00c4cc", "#ff6b9d", "#f1ecff", "#ffffff"] * 2,
        )
    )
    st.markdown(
        f'<div class="marquee" aria-hidden="true"><div class="marquee-track">{isi}</div></div>',
        unsafe_allow_html=True,
    )


def render_bento(cards: Iterable[tuple[str, str, str, str]]) -> None:
    """Empat kartu bento: (angka, judul, isi, kelas warna)."""
    hasil = []
    for index, (angka, judul, isi, kelas) in enumerate(cards):
        hasil.append(
            f'<div class="bento-card {html.escape(kelas)}" style="--urutan:{index}">'
            f'<div class="bento-angka">{html.escape(angka)}</div>'
            f'<h3>{html.escape(judul)}</h3><p>{html.escape(isi)}</p></div>'
        )
    st.markdown(f'<div class="bento">{"".join(hasil)}</div>', unsafe_allow_html=True)
