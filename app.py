"""Entrypoint aplikasi: gaya dimuat sekali, lalu router halaman."""

from __future__ import annotations

from pathlib import Path

# Impor berat dipanaskan di sini agar kunjungan pertama ke halaman Baca Plat
# dan Evaluasi tidak menahan render. Tidak ada logika bisnis di entrypoint.
import cv2  # noqa: F401
import kagglehub  # noqa: F401
import pandas as pd  # noqa: F401
import streamlit as st

from core.pipeline import PipelineParams  # noqa: F401
from ui.components import inject_theme

LOGO_DIR = Path(__file__).resolve().parent / "assets" / "logo"

st.set_page_config(
    page_title="Plat Scanner",
    page_icon=str(LOGO_DIR / "favicon.png"),
    layout="wide",
)

# Animasi masuk hanya untuk render pertama dalam satu sesi.
if "intro_done" not in st.session_state:
    st.session_state["intro_done"] = False

# Satu-satunya injeksi CSS/font. Elemen entrypoint bertahan saat pindah halaman,
# sehingga menu pil melayang dan latar tidak pernah kembali ke gaya bawaan.
inject_theme()

# Logo statis di header sidebar, lengkap dengan favicon sebagai ikon pendamping.
st.logo(
    str(LOGO_DIR / "logo-utama.svg"),
    icon_image=str(LOGO_DIR / "favicon.png"),
    size="large",
)

HALAMAN = [
    st.Page(
        "pages/0_Beranda.py",
        title="Beranda",
        url_path="beranda",
        default=True,
    ),
    st.Page(
        "pages/1_Baca_Plat.py",
        title="Baca Plat",
        url_path="baca-plat",
    ),
    st.Page(
        "pages/2_Evaluasi.py",
        title="Evaluasi",
        url_path="evaluasi",
    ),
    st.Page(
        "pages/3_Dataset.py",
        title="Dataset",
        url_path="dataset",
    ),
    st.Page(
        "pages/4_Uji_Widget.py",
        title="Uji Widget",
        url_path="uji-widget",
        visibility="hidden",
    ),
]

st.navigation(HALAMAN, position="sidebar", expanded=True).run()
