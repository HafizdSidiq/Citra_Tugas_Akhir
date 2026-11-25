"""Beranda aplikasi pengenalan plat nomor kendaraan Indonesia."""

from __future__ import annotations

import streamlit as st

from ui.components import (
    append_history,
    bootstrap,
    intro_class,
    mark_intro_done,
    page_header,
    performance_from_result,
    render_history,
    render_performance_panel,
    render_spotlight,
    render_stat_grid,
)

bootstrap("Beranda · Plat Nomor")

main, perf = st.columns([1.73, 1], gap="large")

with main:
    page_header(
        "Beranda",
        "Pengolahan citra klasik untuk membaca plat nomor Indonesia, langkah demi langkah.",
    )
    render_spotlight(
        "Siap membaca plat",
        "Unggah foto kendaraan atau pilih contoh dari dataset. Setiap tahap bisa dilihat dan diukur.",
    )

    history = list(st.session_state.get("history", []))
    result = st.session_state.get("result")
    if history:
        avg_conf = sum(float(item.get("confidence", 0.0)) for item in history) / len(history)
        stat_values = [
            ("Plat terbaca", f"{len(history)}", f"+{min(len(history), 3)} terakhir", True),
            ("Keyakinan rata-rata", f"{avg_conf:.0f}%", "dari OCR terakhir", avg_conf >= 50),
            (
                "Waktu pemrosesan",
                f"{float(getattr(result, 'timings_ms', {}).get('total', 0.0)):.0f} ms",
                "tahap penuh",
                True,
            ),
        ]
    else:
        stat_values = [
            ("Plat terbaca", "0", "mulai dari sini", False),
            ("Keyakinan rata-rata", "—", "belum ada data", False),
            ("Waktu pemrosesan", "—", "belum ada data", False),
        ]
    render_stat_grid(stat_values)

    st.subheader("Terakhir dibaca")
    render_history(history)
    st.page_link("pages/1_Deteksi.py", label="Mulai baca plat →", icon="🚗")

with perf:
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
        note="Buka halaman Deteksi untuk mengisi panel ini dengan hasil nyata.",
    )

mark_intro_done()
