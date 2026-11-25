"""Halaman Deteksi: unggah/pilih contoh, baca plat, lihat hasil."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.dataset import DatasetError, dataset_status, gallery_items, get_image_bytes
from core.pipeline import process_image
from ui.components import (
    append_history,
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
    render_scan_overlay,
)

bootstrap("Deteksi · Plat Nomor")


main, perf = st.columns([1.73, 1], gap="large")

with main:
    page_header("Deteksi", "Pilih foto kendaraan, lalu biarkan alur klasik membaca platnya.")

    mode = st.radio(
        "Sumber citra",
        ["Unggah foto", "Contoh dari dataset"],
        horizontal=True,
        key="source_mode",
    )
    source_bytes: bytes | None = None
    source_name = ""

    if mode == "Unggah foto":
        uploaded = st.file_uploader(
            "Foto kendaraan (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploaded_photo",
        )
        if uploaded is not None:
            source_bytes = uploaded.getvalue()
            source_name = uploaded.name
            st.image(source_bytes, caption=source_name, width="stretch")
        else:
            render_empty_state(
                "Belum ada foto",
                "Unggah foto kendaraan di atas atau pindah ke Contoh dari dataset.",
            )
    else:
        status = dataset_status()
        if not status["available"]:
            render_failure_card(
                status["message"]
                + " Galeri dinonaktifkan sementara; unggah foto lokal tetap berfungsi."
            )
            st.caption("Koneksi dataset gagal — pilih mode Unggah foto untuk melanjutkan.")
        else:
            items = gallery_items(42, 24)
            labels = [item["label"] for item in items]
            selected = st.selectbox("Pilih contoh", labels, key="gallery_choice")
            chosen = next(item for item in items if item["label"] == selected)
            try:
                source_bytes = get_image_bytes(chosen["image_path"])
                source_name = chosen["label"] + ".jpg"
                st.image(source_bytes, caption=source_name, width="stretch")
            except DatasetError as exc:
                render_failure_card(str(exc))

            with st.expander("Lihat 24 contoh dari dataset", expanded=False):
                cols = st.columns(6)
                for idx, item in enumerate(items):
                    with cols[idx % 6]:
                        try:
                            st.image(
                                get_image_bytes(item["image_path"]),
                                caption=item["label"],
                                width="stretch",
                            )
                        except DatasetError:
                            st.caption(f"{item['label']}: gagal dimuat")

    params = parameter_widgets()
    if source_bytes is not None:
        st.session_state["source_bytes"] = source_bytes
        st.session_state["source_name"] = source_name

    read = st.button("Baca plat", type="primary", width="stretch")
    if read:
        if source_bytes is None:
            render_failure_card(
                "Belum ada citra yang dipilih. Unggah foto atau pilih contoh dataset dulu, "
                "lalu tekan Baca plat kembali."
            )
        else:
            render_scan_overlay()
            with st.spinner("Menjalankan delapan tahap pengolahan citra…"):
                result = process_image(source_bytes, params)
            st.session_state["result"] = result
            append_history(result.text or "?", result.confidence)

    result = st.session_state.get("result")
    if result is not None:
        st.subheader("Hasil pembacaan")
        render_image_card(
            result.annotated_image,
            "Citra dengan kotak plat terdeteksi",
            css_class="result-card",
        )
        if result.text:
            render_result_text(result.text)
            st.caption(
                f"Keyakinan OCR {result.confidence:.0f}% · "
                f"{len(result.plates)} plat terdeteksi"
            )
        else:
            render_failure_card(
                "Teks plat belum terbaca. Coba foto yang lebih dekat dan tajam, "
                "atau sesuaikan ambang Canny pada Parameter pengolahan."
            )

        for warning in result.warnings:
            render_failure_card(warning)

        if result.plates:
            rows = [
                {
                    "Plat": plate.index,
                    "Teks": plate.text or "—",
                    "Keyakinan (%)": plate.confidence,
                    "Pola valid": "Ya" if plate.valid_pattern else "Tidak",
                    "Binisasi": plate.threshold_used,
                    "Karakter": len(plate.char_boxes),
                }
                for plate in result.plates
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander("Waktu per tahap", expanded=False):
            timings = pd.DataFrame(
                [
                    {"Tahap": name, "Waktu (ms)": value}
                    for name, value in result.timings_ms.items()
                ]
            )
            st.dataframe(timings, hide_index=True, width="stretch")

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
        note="Panel ini memakai hasil terakhir. Teks yang tidak cocok pola tetap ditampilkan.",
    )

mark_intro_done()
