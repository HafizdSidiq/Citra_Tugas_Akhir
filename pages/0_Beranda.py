"""Beranda: identitas Plat Scanner, jendela pemindai, dan kartu bento."""

from __future__ import annotations

import streamlit as st

from ui.components import (
    mark_intro_done,
    render_bento,
    render_hero_stage,
    render_identity_header,
    render_logo,
)

with st.container(border=True, key="kartu_beranda"):
    kiri, kanan = st.columns([1.05, 1], gap="large")

    with kiri:
        render_logo(animated=True, ukuran=160)
        render_identity_header()
        st.page_link(
            "pages/1_Baca_Plat.py",
            label="Baca plat",
        )
        st.caption(
            "Tahap pengolahan ditampilkan satu per satu agar hasil bisa diaudit, "
            "bukan sekadar teks dari kotak hitam."
        )

    with kanan:
        render_hero_stage()

with st.container(border=True, key="kartu_bento"):
    st.subheader("Sekilas proyek")
    render_bento(
        [
            (
                "8",
                "Tahap pengolahan",
                "Resize, CLAHE, Canny, Kontur, Perspektif, Segmentasi, OCR, Hasil.",
                "bento-ungu",
            ),
            (
                "1000",
                "Citra deteksi",
                "Contoh dari split test dataset publik yang dipakai sebagai ilustrasi.",
                "",
            ),
            (
                "1906",
                "Crop plat",
                "Kotak plat yang dipakai untuk menguji lokalisasi dan bacaan karakter.",
                "bento-kuning",
            ),
            (
                "0",
                "Deep learning",
                "Tanpa PyTorch atau TensorFlow; seluruh alur memakai OpenCV dan Tesseract.",
                "bento-tosca",
            ),
        ]
    )

mark_intro_done()
