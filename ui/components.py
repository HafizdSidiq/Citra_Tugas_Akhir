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
LOGO_DIR = BASE_DIR.parent / "assets" / "logo"

WARNA_STAT = ("stat-putih", "stat-ungu", "stat-kuning", "stat-tosca", "stat-pink")

# Font dimuat lewat <link> di entrypoint, bukan @import di dalam CSS, agar
# tidak terjadi FOUT ketika halaman berganti.
FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;700'
    '&family=Unbounded:wght@600;800&display=swap">'
)


def inject_theme() -> None:
    """Suntik font dan gaya sekali dari entrypoint, sebelum navigasi.

    Memakai st.html, bukan st.markdown, agar seluruh CSS tidak dipotong oleh
    parser markdown saat string HTML panjang disuntikkan.
    """
    st.html(FONT_LINKS + f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>")


def bootstrap() -> None:
    """Alias lama; kini hanya meneruskan ke inject_theme(), tanpa session_state."""
    inject_theme()


def mark_intro_done() -> None:
    """Tandai animasi introduksi selesai setelah render pertama."""
    st.session_state["intro_done"] = True


def intro_class() -> str:
    """Kelas animasi hanya untuk render pertama."""
    return "intro-only" if not st.session_state.get("intro_done", True) else ""


def page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "Plat Scanner · Pengolahan Citra Digital",
) -> None:
    st.markdown(
        f'<div class="{intro_class()}"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_identity_header() -> None:
    """Identitas Plat Scanner untuk Beranda, mengikuti desain sampul."""
    judul_resmi = (
        "Pengenalan Plat Nomor Kendaraan Indonesia dengan Pengolahan Citra "
        "Klasik dan Tesseract OCR"
    )
    st.markdown(
        f'<div class="{intro_class()} hero-identitas">'
        '<div class="eyebrow">Pengolahan Citra Digital · Tugas Akhir</div>'
        '<h1 class="hero-judul"><span class="baris-1">PLAT</span>'
        '<span class="baris-2">SCANNER</span></h1>'
        '<p class="hero-tagline">Plat terbaca, langkahnya terlihat.</p>'
        f'<p class="hero-judul-resmi">{html.escape(judul_resmi)}</p>'
        '<p class="hero-penjelasan">Unggah foto, atur parameter, dan periksa '
        'setiap tahapnya.</p>'
        '<div class="hero-pil">'
        '<span class="pil-opencv">OpenCV</span>'
        '<span class="pil-tesseract">Tesseract</span>'
        '<span class="pil-streamlit">Streamlit</span>'
        '</div></div>',
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


def render_logo(animated: bool, ukuran: int = 160) -> None:
    """Tampilkan logo SCAN dari berkas aset; SVG disuntik lewat iframe."""
    lebar = max(120, min(220, int(ukuran)))
    if animated:
        isi = (LOGO_DIR / "logo-animasi-inline.html").read_text(encoding="utf-8")
        isi = isi.replace('aria-label="Logo SCAN animasi"', 'aria-label="Logo SCAN"')
        tambahan = (
            "<style>html,body{margin:0;background:transparent;overflow:hidden}"
            f".wadah-logo{{width:{lebar}px;--scn-fg:#14112b;--scn-accent:#00c4cc}}"
            "</style>"
        )
    else:
        isi = (LOGO_DIR / "logo-utama.svg").read_text(encoding="utf-8")
        isi = isi.replace("#0E1B2B", "#14112b").replace("#0A9F92", "#00c4cc")
        tambahan = (
            "<style>html,body{margin:0;background:transparent;overflow:hidden}"
            f".logo-statis{{width:{lebar}px}}"
            ".logo-statis svg{display:block;width:100%;height:auto}</style>"
        )
        isi = f'<div class="logo-statis">{isi}</div>'
    st.iframe(isi + tambahan, width="stretch", height=lebar + 4)


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


def render_plate_headline(
    text: str, confidence: float, n_plates: int, skor: float | None = None
) -> None:
    """Teks plat besar; kandidat utama diberi label jelas."""
    if not text:
        return
    spans = "".join(
        f'<span style="animation-delay:{i * 90}ms">{html.escape(char)}</span>'
        for i, char in enumerate(text)
    )
    skor_pil = (
        f'<span class="pil-keyakinan">Skor {float(skor):.0f}</span> ' if skor is not None else ""
    )
    st.markdown(
        f'<span style="font-size:13px;font-weight:700">Hasil utama</span>'
        '<div class="result-headline">'
        f'<div class="result-text">{spans}</div>'
        f'<span class="pil-keyakinan">Keyakinan {float(confidence):.0f}%</span> '
        f'{skor_pil}'
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
            low = st.slider(
                "Ambang Canny bawah",
                20,
                120,
                70,
                5,
                key="canny_low",
                help="Batas tepi lemah. Turunkan bila plat kecil tidak terdeteksi.",
            )
            high = st.slider(
                "Ambang Canny atas",
                60,
                260,
                210,
                10,
                key="canny_high",
                help="Batas tepi kuat. Naikkan bila terlalu banyak kontur palsu.",
            )
            min_area = st.slider(
                "Luas minimum (× citra)",
                0.0002,
                0.005,
                0.0003,
                0.0001,
                format="%.4f",
                key="min_area",
                help="Kotak lebih kecil dari nilai ini diabaikan sebagai plat.",
            )
            max_area = st.slider(
                "Luas maksimum (× citra)",
                0.05,
                0.60,
                0.25,
                0.05,
                format="%.2f",
                key="max_area",
                help="Kotak lebih besar dari ini dianggap bingkai/foto, bukan plat.",
            )
        with kanan:
            min_score = st.slider(
                "Ambang skor hasil",
                0,
                90,
                55,
                5,
                key="min_score",
                help="Bacaan di bawah skor ini dibuang agar kotak dan teks palsu berkurang.",
            )
            aspect_min = st.slider(
                "Rasio minimum",
                1.4,
                3.0,
                1.4,
                0.1,
                key="aspect_min",
                help="Rasio lebar terhadap tinggi terkecil yang diterima sebagai plat.",
            )
            aspect_max = st.slider(
                "Rasio maksimum",
                2.8,
                4.2,
                3.8,
                0.1,
                key="aspect_max",
                help="Rasio lebar terhadap tinggi terbesar yang diterima sebagai plat.",
            )
            use_clahe = st.checkbox(
                "Gunakan CLAHE",
                value=True,
                key="use_clahe",
                help="Meningkatkan kontras pada foto gelap atau berkabut.",
            )
            binarization = st.selectbox(
                "Binerisasi",
                ["auto", "otsu", "adaptive"],
                index=0,
                key="binarization",
                help="auto mencoba Otsu dan adaptif, lalu memilih skor OCR tertinggi.",
            )
    high = max(high, low + 10)
    aspect_max = max(aspect_max, aspect_min + 0.2)
    return PipelineParams(
        canny_low=low,
        canny_high=high,
        min_area_ratio=float(min_area),
        max_area_ratio=float(max_area),
        min_score=float(min_score),
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
# Beranda: panggung simulasi dan bento (CSS murni, tanpa JS kecuali
# iframe yang memang diisolasi dari markdown).
# --------------------------------------------------------------------------
TAHAP_SIMULASI = [
    ("1", "Resize", "#241f47"),
    ("2", "CLAHE", "#241f47"),
    ("3", "Canny", "#241f47"),
    ("4", "Kontur", "#241f47"),
    ("5", "Perspektif", "#241f47"),
    ("6", "Segmentasi", "#241f47"),
    ("7", "OCR", "#ffd23f"),
    ("8", "Hasil", "#00c4cc"),
]


def render_hero_stage() -> None:
    """Jendela pemindai gelap dengan delapan tahap sebagai pil kecil."""
    pil = []
    for nomor, nama, warna in TAHAP_SIMULASI:
        kelas = "pil-tahap pil-gelap"
        if nomor == "7":
            kelas = "pil-tahap pil-terang"
        elif nomor == "8":
            kelas = "pil-tahap pil-terang"
        pil.append(
            f'<span class="{kelas}" style="background:{warna}">'
            f'<b>{nomor}</b> {html.escape(nama)}</span>'
        )
    css = """
      html,body{margin:0;background:transparent;font-family:Figtree,system-ui,sans-serif}
      .pemindai{position:relative;width:100%;height:300px;border-radius:36px;overflow:hidden;
        background-color:#1b1736;
        background-image:linear-gradient(rgba(245,242,255,.07) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(245,242,255,.07) 1px,transparent 1px);
        background-size:34px 34px;
        box-shadow:0 34px 64px -34px rgba(20,17,43,.75)}
      .siku{position:absolute;width:54px;height:54px;border:0 solid #00c4cc;pointer-events:none}
      .siku.tl{top:18px;left:18px;border-top-width:6px;border-left-width:6px;border-top-left-radius:16px}
      .siku.tr{top:18px;right:18px;border-top-width:6px;border-right-width:6px;border-top-right-radius:16px}
      .siku.bl{bottom:18px;left:18px;border-bottom-width:6px;border-left-width:6px;border-bottom-left-radius:16px}
      .siku.br{bottom:18px;right:18px;border-bottom-width:6px;border-right-width:6px;border-bottom-right-radius:16px}
      .target{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
        padding:10px 14px;border:4px solid #ffd23f;border-radius:26px}
      .label{position:absolute;left:10px;top:-16px;padding:4px 10px;border-radius:99px;
        background:#ffd23f;color:#14112b;font-size:10px;font-weight:800;letter-spacing:.08em;white-space:nowrap}
      .plat{display:block;padding:14px 26px;border-radius:16px;background:#fff;color:#14112b;
        font:800 26px "Unbounded","Figtree",sans-serif;letter-spacing:.08em;white-space:nowrap}
      .garis-pindai{position:absolute;left:6%;right:6%;height:3px;top:14%;border-radius:99px;
        background:#ff6b9d;box-shadow:0 0 16px rgba(255,107,157,.9);pointer-events:none;
        animation:pindai 4.5s ease-in-out infinite alternate}
      @keyframes pindai{from{top:14%}to{top:86%}}
      .keterangan{position:absolute;left:16px;bottom:14px;color:#b4aed1;font-size:12px;font-weight:700}
      .tahap{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:14px}
      .pil-tahap{padding:7px 12px;border-radius:99px;font-size:12px;font-weight:700;white-space:nowrap}
      .pil-tahap b{margin-right:4px}
      .pil-gelap{color:#f5f2ff}
      .pil-terang{color:#14112b}
      @media(prefers-reduced-motion:reduce){.garis-pindai{animation:none!important;top:50%;opacity:.5}}
      @media(max-width:560px){.pemindai{height:260px}.plat{font-size:20px;padding:12px 18px}
        .siku{width:40px;height:40px}.keterangan{font-size:11px}.pil-tahap{font-size:11px;padding:6px 9px}}
    """
    st.iframe(
        FONT_LINKS + f"<style>{css}</style>"
        '<div class="pemindai">'
        '<i class="siku tl" aria-hidden="true"></i><i class="siku tr" aria-hidden="true"></i>'
        '<i class="siku bl" aria-hidden="true"></i><i class="siku br" aria-hidden="true"></i>'
        '<div class="target"><span class="label">PLAT TERDETEKSI</span>'
        '<span class="plat">B1234ABC</span></div>'
        '<i class="garis-pindai" aria-hidden="true"></i>'
        '<span class="keterangan">Simulasi ilustratif, bukan hasil OCR sungguhan</span>'
        '</div><div class="tahap">' + "".join(pil) + "</div>",
        width="stretch",
        height=440,
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
