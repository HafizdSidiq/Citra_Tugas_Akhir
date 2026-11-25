"""Beranda: hero simulasi tahap, marquee kondisi, dan kartu bento."""

from __future__ import annotations

import streamlit as st

from ui.components import (
    bootstrap,
    mark_intro_done,
    page_header,
    render_bento,
    render_hero_stage,
    render_marquee,
)

bootstrap()

with st.container(border=True, key="kartu_beranda"):
    kiri, kanan = st.columns([1.05, 1], gap="large")

    with kiri:
        page_header(
            "Plat terbaca, langkahnya terlihat.",
            "Baca plat nomor Indonesia dengan pengolahan citra klasik. Unggah foto, "
            "atur parameter, dan periksa setiap tahapnya.",
        )
        st.page_link(
            "pages/1_Baca_Plat.py",
            label="Baca plat",
            icon="🚗",
        )
        st.caption(
            "Tahap pengolahan ditampilkan satu per satu agar hasil bisa diaudit, "
            "bukan sekadar teks dari kotak hitam."
        )

    with kanan:
        render_hero_stage()
        st.caption(
            "Simulasi ilustratif delapan tahap. Bukan hasil OCR sungguhan; "
            "buka Baca Plat untuk memproses foto Anda."
        )

render_marquee(["Siang", "Malam", "Miring", "Motor", "Mobil"])

with st.container(border=True, key="kartu_bento"):
    st.subheader("Sekilas proyek")
    render_bento(
        [
            (
                "8",
                "Tahap pengolahan",
                "Perkecil, abu-abu, CLAHE, Canny, kontur, binerisasi, segmentasi, OCR.",
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
