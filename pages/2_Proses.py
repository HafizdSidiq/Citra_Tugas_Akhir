"""Halaman Proses: tampilkan citra tiap tahap secara berdampingan."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.pipeline import process_image
from ui.components import (
    bootstrap,
    mark_intro_done,
    page_header,
    parameter_widgets,
    performance_from_result,
    render_empty_state,
    render_failure_card,
    render_image_card,
    render_performance_panel,
    render_result_text,
)

bootstrap("Proses · Plat Nomor")

STAGE_ORDER = [
    ("original", "0. Citra asli"),
    ("resize", "1. Resize lebar maks 960 px"),
    ("grayscale", "2a. Grayscale"),
    ("enhanced", "2b. CLAHE + bilateral"),
    ("edges", "3. Canny + closing"),
    ("contours", "4. Kontur persegi panjang"),
    ("rectified_1", "5a. Koreksi perspektif"),
    ("otsu_1", "5b. Binarisasi Otsu"),
    ("adaptive_1", "5c. Binarisasi adaptif"),
    ("characters_1", "6. Segmentasi karakter"),
    ("annotated", "8. Citra berkotak"),
]

main, perf = st.columns([1.73, 1], gap="large")

with main:
    page_header("Proses", "Lihat setiap tahap pengolahan dan ukur waktunya.")

    source_bytes = st.session_state.get("source_bytes")
    source_name = st.session_state.get("source_name", "")
    if source_bytes is None:
        uploaded = st.file_uploader(
            "Unggah foto untuk melihat prosesnya",
            type=["jpg", "jpeg", "png", "webp"],
            key="process_upload",
        )
        if uploaded is not None:
            source_bytes = uploaded.getvalue()
            source_name = uploaded.name
            st.session_state["source_bytes"] = source_bytes
            st.session_state["source_name"] = source_name
        else:
            render_empty_state(
                "Belum ada citra untuk diproses",
                "Buka halaman Deteksi untuk memilih foto, atau unggah langsung di sini.",
            )
            st.page_link("pages/1_Deteksi.py", label="Pergi ke Deteksi →", icon="🚗")
    else:
        st.caption(f"Sumber: {source_name or 'citra dari sesi Deteksi'}")

    params = parameter_widgets()

    result = None
    if source_bytes is not None:
        with st.spinner("Menjalankan tahapan…"):
            result = process_image(source_bytes, params)
        st.session_state["result"] = result

        if result.text:
            st.subheader("Teks terbaca")
            render_result_text(result.text)
        for warning in result.warnings:
            render_failure_card(warning)

        st.subheader("Citra antar tahap")
        available = [(key, label) for key, label in STAGE_ORDER if key in result.stages]
        for idx in range(0, len(available), 2):
            left, right = st.columns(2)
            with left:
                key, label = available[idx]
                render_image_card(result.stages[key], label, delay_ms=idx * 80)
            if idx + 1 < len(available):
                with right:
                    key, label = available[idx + 1]
                    render_image_card(result.stages[key], label, delay_ms=(idx + 1) * 80)

        with st.expander("Waktu per tahap", expanded=True):
            timing_rows = [
                {"Tahap": name, "Waktu (ms)": value}
                for name, value in result.timings_ms.items()
            ]
            st.dataframe(
                pd.DataFrame(timing_rows), hide_index=True, width="stretch"
            )

with perf:
    result = st.session_state.get("result")
    metrics = (
        performance_from_result(result)
        if result is not None
        else [
            ("Keyakinan OCR", 0.0, "—"),
            ("Plat terlokalisasi", 0.0, "—"),
            ("Kecepatan", 0.0, "—"),
        ]
    )
    render_performance_panel(
        metrics,
        note="Kartu tahap muncul bergantian 80 ms. Geser parameter untuk membandingkan hasil.",
    )

mark_intro_done()
