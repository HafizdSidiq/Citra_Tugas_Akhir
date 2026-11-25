"""Halaman Evaluasi: metrik lokalisasi, OCR, skenario, dan waktu tahap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.dataset import dataset_image_ids
from core.evaluation import predictions_from_csv, predictions_from_dataset
from core.metrics import evaluate
from core.pipeline import PipelineParams
from ui.components import (
    bootstrap,
    mark_intro_done,
    page_header,
    parameter_widgets,
    render_empty_state,
    render_failure_card,
    render_performance_panel,
    render_stat_grid,
)

bootstrap("Evaluasi · Plat Nomor")

DEFAULT_REFERENCE = Path("docs/acuan_ocr.csv")
CONDITIONS = [
    "",
    "siang",
    "malam",
    "bayangan",
    "tegak",
    "miring",
    "jauh",
    "mobil",
    "motor",
    "clahe",
    "tanpa_clahe",
    "otsu",
    "adaptif",
]

main, perf = st.columns([1.73, 1], gap="large")
eval_result = None

with main:
    page_header(
        "Evaluasi",
        "Ukur lokalisasi dan bacaan OCR hanya pada data uji yang sudah ditandai.",
    )

    st.subheader("1. Acuan OCR")
    st.caption(
        "Teks plat diketik manual. Kotak berasal dari label YOLO dataset; "
        "kolom condition dipakai untuk mengelompokkan skenario."
    )
    upload_ref = st.file_uploader(
        "Ganti CSV acuan (opsional)", type=["csv"], key="reference_upload"
    )
    if upload_ref is not None:
        reference_df = pd.read_csv(upload_ref, dtype=str, keep_default_na=False)
    elif DEFAULT_REFERENCE.exists():
        reference_df = pd.read_csv(DEFAULT_REFERENCE, dtype=str, keep_default_na=False)
    else:
        reference_df = pd.DataFrame(columns=["image_id", "text", "box", "condition"])

    needed = {"image_id", "text", "box", "condition"}
    if not needed.issubset(set(reference_df.columns)):
        render_failure_card(
            "CSV acuan harus memiliki kolom image_id, text, box, dan condition."
        )
        reference_df = pd.DataFrame(columns=sorted(needed))

    edited = st.data_editor(
        reference_df,
        column_config={
            "image_id": st.column_config.TextColumn("Citra", disabled=True),
            "box": st.column_config.TextColumn("Kotak acuan (px)", disabled=True),
            "text": st.column_config.TextColumn(
                "Teks plat acuan", help="Ketik manual dari citra; kosongkan bila belum dicek."
            ),
            "condition": st.column_config.SelectboxColumn(
                "Kondisi", options=CONDITIONS, help="Contoh: siang, malam, miring, otsu."
            ),
        },
        hide_index=True,
        width="stretch",
        height=260,
    )
    filled = int((edited["text"].astype(str).str.strip() != "").sum())
    st.caption(
        f"{filled} dari {len(edited)} baris sudah berisi teks acuan. "
        "Baris kosong tetap dihitung sebagai miss bila masuk data uji."
    )
    st.download_button(
        "Unduh acuan yang sudah diisi",
        data=edited.to_csv(index=False).encode("utf-8-sig"),
        file_name="acuan_ocr_terisi.csv",
        mime="text/csv",
        disabled=len(edited) == 0,
    )

    st.subheader("2. Prediksi")
    pred_source = st.radio(
        "Sumber prediksi",
        ["Jalankan pipeline", "CSV prediksi"],
        horizontal=True,
        key="prediction_source",
    )
    predictions: dict[str, list[dict[str, object]]] = {}
    base_params = PipelineParams()

    if pred_source == "CSV prediksi":
        upload_pred = st.file_uploader(
            "CSV prediksi (image_id, text, box, confidence, condition)",
            type=["csv"],
            key="prediction_upload",
        )
        if upload_pred is not None:
            try:
                predictions = predictions_from_csv(upload_pred)
                st.caption(f"{len(predictions)} citra memiliki prediksi.")
            except ValueError as exc:
                render_failure_card(str(exc))
        else:
            render_empty_state(
                "Belum ada prediksi",
                "Unggah CSV prediksi atau pilih Jalankan pipeline.",
            )
    else:
        base_params = parameter_widgets()
        limit = st.slider("Jumlah citra yang diproses", 4, 24, 12, 4, key="eval_limit")
        if st.button("Jalankan prediksi", type="primary"):
            image_ids = dataset_image_ids(42, limit)
            params_key = tuple(sorted(vars(base_params).items()))
            cache_key = (params_key, limit)
            if st.session_state.get("eval_cache_key") != cache_key:
                with st.spinner(f"Memproses {limit} citra dari dataset…"):
                    try:
                        predictions = predictions_from_dataset(image_ids, base_params, limit=limit)
                        st.session_state["eval_predictions"] = predictions
                        st.session_state["eval_cache_key"] = cache_key
                    except Exception as exc:
                        render_failure_card(
                            f"Prediksi gagal dijalankan: {exc}. Cek koneksi Kaggle atau unggah CSV prediksi."
                        )
            else:
                predictions = dict(st.session_state.get("eval_predictions", {}))
        else:
            predictions = dict(st.session_state.get("eval_predictions", {}))
            if predictions:
                st.caption("Memakai prediksi dari sesi ini. Geser parameter lalu jalankan ulang untuk membandingkan.")

    st.subheader("3. Metrik")
    predicted_ids = {str(key) for key in predictions}
    reference_used = edited[edited["image_id"].astype(str).isin(predicted_ids)].copy()
    if not predictions or reference_used.empty:
        render_empty_state(
            "Metrik belum tersedia",
            "Teks acuan dan prediksi harus menyentuh citra yang sama. Isi salah satu baris "
            "lalu jalankan prediksi atau unggah CSV prediksi.",
        )
    else:
        result = evaluate(reference_used, predictions, iou_threshold=0.5, seed=42, test_fraction=0.2)
        eval_result = result
        st.caption(
            f"Split seed 42 pada {len(reference_used)} baris acuan yang diprediksi: "
            f"{len(result.train_ids)} baris latih, {result.n} baris uji. "
            "Angka di bawah hanya berlaku untuk baris uji ini."
        )

        render_stat_grid(
            [
                (
                    "IoU rata-rata",
                    f"{result.localization['iou_rata_rata']:.2f}",
                    f"ambang {result.localization['ambang_iou']:.1f}",
                    result.localization["iou_rata_rata"] >= 0.5,
                ),
                (
                    "Akurasi plat",
                    f"{result.plate_metrics['akurasi_plat'] * 100:.0f}%",
                    "teks sama persis",
                    result.plate_metrics["akurasi_plat"] >= 0.5,
                ),
                (
                    "Akurasi karakter",
                    f"{result.plate_metrics['akurasi_karakter'] * 100:.0f}%",
                    f"{result.char_metrics['benar']}/{result.char_metrics['total']} karakter",
                    result.plate_metrics["akurasi_karakter"] >= 0.5,
                ),
            ]
        )

        for warning in result.warnings:
            render_failure_card(warning)

        left, right = st.columns(2)
        with left:
            st.markdown("**Lokalisasi**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metrik": "Presisi", "Nilai": f"{result.localization['presisi']:.3f}"},
                        {"Metrik": "Recall", "Nilai": f"{result.localization['recall']:.3f}"},
                        {"Metrik": "F1", "Nilai": f"{result.localization['f1']:.3f}"},
                        {"Metrik": "Tingkat deteksi", "Nilai": f"{result.localization['deteksi']:.3f}"},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        with right:
            st.markdown("**Confusion matrix teks plat**")
            st.dataframe(result.confusion, width="stretch", height=220)

        with st.expander("Skenario siang/malam/bayangan, tegak/miring/jauh, dll.", expanded=False):
            st.dataframe(result.scenarios, hide_index=True, width="stretch")

        with st.expander("Ringkasan per kondisi terisi", expanded=False):
            if result.per_condition.empty:
                st.caption("Belum ada kolom condition yang diisi.")
            else:
                st.dataframe(result.per_condition, hide_index=True, width="stretch")

        with st.expander("Waktu rata-rata per tahap", expanded=False):
            st.dataframe(result.timings, hide_index=True, width="stretch")

    st.subheader("4. Perbandingan varian")
    st.caption(
        "Empat varian dijalankan pada citra yang sama: CLAHE aktif/nonaktif dan "
        "Otsu/adaptif. Kolom condition acuan dikosongkan agar label varian yang dipakai."
    )
    compare_limit = st.slider(
        "Citra per varian", 4, 12, 6, 2, key="eval_compare_limit"
    )
    if st.button("Bandingkan varian", type="primary"):
        variants = [
            ("CLAHE aktif", "clahe", True, "auto"),
            ("Tanpa CLAHE", "tanpa_clahe", False, "auto"),
            ("Otsu", "otsu", True, "otsu"),
            ("Adaptif", "adaptif", True, "adaptive"),
        ]
        neutral_reference = edited[
            edited["image_id"].astype(str).isin(set(dataset_image_ids(42, compare_limit)))
        ].copy()
        neutral_reference["condition"] = ""
        comparison_rows: list[dict[str, object]] = []
        try:
            for display, label, use_clahe, binary in variants:
                variant_params = PipelineParams(
                    **{**vars(base_params), "use_clahe": use_clahe, "binarization": binary}
                )
                variant_ids = dataset_image_ids(42, compare_limit)
                with st.spinner(f"Menjalankan {display}…"):
                    variant_predictions = predictions_from_dataset(
                        variant_ids,
                        variant_params,
                        conditions={name: label for name in variant_ids},
                        limit=compare_limit,
                    )
                variant_result = evaluate(
                    neutral_reference,
                    variant_predictions,
                    iou_threshold=0.5,
                    seed=42,
                    test_fraction=0.2,
                )
                total_values = [
                    float(item.get("timings_ms", {}).get("total", 0.0))
                    for items in variant_predictions.values()
                    for item in items[:1]
                ]
                comparison_rows.append(
                    {
                        "Varian": display,
                        "IoU": round(variant_result.localization["iou_rata_rata"], 3),
                        "Akurasi plat": round(variant_result.plate_metrics["akurasi_plat"], 3),
                        "Akurasi karakter": round(
                            variant_result.plate_metrics["akurasi_karakter"], 3
                        ),
                        "Waktu total (ms)": round(
                            sum(total_values) / len(total_values) if total_values else 0.0,
                            1,
                        ),
                    }
                )
            st.session_state["eval_comparison_rows"] = comparison_rows
        except Exception as exc:
            render_failure_card(
                f"Perbandingan gagal: {exc}. Cek koneksi Kaggle atau unggah CSV prediksi."
            )

    comparison_rows = list(st.session_state.get("eval_comparison_rows", []))
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")

with perf:
    metrics = (
        [
            ("IoU rata-rata", eval_result.localization["iou_rata_rata"] * 100, f"{eval_result.localization['iou_rata_rata']:.2f}"),
            ("Akurasi plat", eval_result.plate_metrics["akurasi_plat"] * 100, f"{eval_result.plate_metrics['akurasi_plat'] * 100:.0f}%"),
            ("Presisi", eval_result.localization["presisi"] * 100, f"{eval_result.localization['presisi']:.0f}%"),
        ]
        if eval_result is not None
        else [
            ("IoU rata-rata", 0.0, "—"),
            ("Akurasi plat", 0.0, "—"),
            ("Presisi", 0.0, "—"),
        ]
    )
    render_performance_panel(
        metrics,
        note="Panel kanan memakai hasil evaluasi terakhir pada baris uji. Tidak ada klaim di luar data uji.",
    )

mark_intro_done()
