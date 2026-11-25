"""Entrypoint aplikasi: hanya router halaman."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Plat Nomor",
    page_icon="🚗",
    layout="wide",
)

HALAMAN = [
    st.Page(
        "pages/0_Beranda.py",
        title="Beranda",
        icon="🏠",
        url_path="beranda",
        default=True,
    ),
    st.Page(
        "pages/1_Baca_Plat.py",
        title="Baca Plat",
        icon="🚗",
        url_path="baca-plat",
    ),
    st.Page(
        "pages/2_Evaluasi.py",
        title="Evaluasi",
        icon="📊",
        url_path="evaluasi",
    ),
    st.Page(
        "pages/3_Dataset.py",
        title="Dataset",
        icon="🗂️",
        url_path="dataset",
    ),
    st.Page(
        "pages/4_Uji_Widget.py",
        title="Uji Widget",
        icon="🧪",
        url_path="uji-widget",
        visibility="hidden",
    ),
]

st.navigation(HALAMAN, position="sidebar", expanded=True).run()
