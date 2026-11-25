"""Evaluasi: empat tab berurutan dari acuan hingga perbandingan varian."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from core.dataset import dataset_image_ids
from core.evaluation import predictions_from_csv, predictions_from_dataset
from core.metrics import evaluate
from core.pipeline import PipelineParams
from ui.components import (
    mark_intro_done,
    page_header,
    parameter_widgets,
    render_empty_state,
    render_failure_card,
    render_info_card,
    render_summary_card,
    render_stat_grid,
)

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


def status_chip(teks: str, siap: bool) -> None:
    """Tandai status sebuah tab dengan kontras yang cukup."""
    kelas = "step-status step-siap" if siap else "step-status step-belum"
    simbol = "✓" if siap else "○"
    st.markdown(
        f'<div class="{kelas}"><strong>{simbol}</strong> {html.escape(teks)}</div>',
        unsafe_allow_html=True,
    )


def terjemah_pesan(pesan: str) -> str:
    """Samakan istilah luaran core dengan bahasa antarmuka."""
    return str(pesan).replace("miss", "luput")


with st.container(border=True, key="kartu_evaluasi"):
    page_header(
        "Evaluasi",
        "Ukur lokalisasi dan bacaan OCR hanya pada data uji yang sudah ditandai.",
    )

    tab_acuan, tab_prediksi, tab_hasil, tab_banding = st.tabs(
        [
            "1 Acuan",
            "2 Prediksi",
            "3 Hasil",
            "4 Bandingkan varian",
        ]
    )

    edited = None
    filled = 0
    predictions: dict[str, list[dict[str, object]]] = {}
    base_params = PipelineParams()

    # -------------------------------------------------- 1. Acuan
    with tab_acuan:
        status_chip(
            "Unggah atau periksa CSV acuan, lalu isi teks platnya.",
            bool(st.session_state.get("acuan_terisi", False)),
        )
        st.caption(
            "Isi kolom **Teks plat acuan** untuk citra yang ingin dinilai, lalu pilih "
            "**Kondisi** bila diketahui. Nama citra dan kotak sudah disiapkan dari "
            "dataset; Anda bisa mengganti CSV atau mengunduh hasilnya di bawah tabel."
        )
        unggah_acuan = st.file_uploader(
            "Ganti CSV acuan (opsional)",
            type=["csv"],
            key="unggah_acuan",
        )
        if unggah_acuan is not None:
            reference_df = pd.read_csv(
                unggah_acuan, dtype=str, keep_default_na=False
            )
            sumber_id = hashlib.sha256(unggah_acuan.getvalue()).hexdigest()[:10]
        elif DEFAULT_REFERENCE.exists():
            reference_df = pd.read_csv(
                DEFAULT_REFERENCE, dtype=str, keep_default_na=False
            )
            sumber_id = "bawaan"
        else:
            reference_df = pd.DataFrame(
                columns=["image_id", "text", "box", "condition"]
            )
            sumber_id = "kosong"

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
                "box": None,
                "text": st.column_config.TextColumn(
                    "Teks plat acuan",
                    help="Ketik manual dari citra; kosongkan bila belum dicek.",
                ),
                "condition": st.column_config.SelectboxColumn(
                    "Kondisi",
                    options=CONDITIONS,
                    help="Contoh: siang, malam, miring, otsu.",
                ),
            },
            hide_index=True,
            width="stretch",
            height=260,
            key=f"editor_acuan_{sumber_id}",
        )
        filled = int((edited["text"].astype(str).str.strip() != "").sum())
        st.session_state["acuan_terisi"] = filled > 0
        st.caption(
            f"{filled} dari {len(edited)} baris acuan terisi. "
            "Baris kosong tetap dihitung sebagai luput bila masuk data uji."
        )
        st.download_button(
            "Unduh acuan yang sudah diisi",
            data=edited.to_csv(index=False).encode("utf-8-sig"),
            file_name="acuan_ocr_terisi.csv",
            mime="text/csv",
            disabled=len(edited) == 0,
            width="stretch",
        )

    # ------------------------------------------------ 2. Prediksi
    with tab_prediksi:
        status_chip(
            "Jalankan pipeline atau unggah CSV prediksi.",
            bool(st.session_state.get("eval_predictions")),
        )
        pred_source = st.radio(
            "Sumber prediksi",
            ["Jalankan pipeline", "CSV prediksi"],
            horizontal=True,
            key="prediction_source",
        )
        if pred_source == "CSV prediksi":
            unggah_pred = st.file_uploader(
                "CSV prediksi (image_id, text, box, confidence, condition)",
                type=["csv"],
                key="unggah_prediksi",
            )
            if unggah_pred is not None:
                try:
                    predictions = predictions_from_csv(unggah_pred)
                    st.session_state["eval_predictions"] = predictions
                    st.session_state["eval_cache_key"] = None
                    st.caption(f"{len(predictions)} citra memiliki prediksi.")
                except ValueError as exc:
                    render_failure_card(str(exc))
            else:
                render_empty_state(
                    "Belum ada prediksi",
                    "Unggah CSV prediksi atau pilih Jalankan pipeline.",
                )
                predictions = dict(st.session_state.get("eval_predictions", {}))
        else:
            base_params = parameter_widgets()
            batas = st.slider(
                "Jumlah citra yang diproses", 4, 24, 12, 4, key="eval_limit"
            )
            if st.button("Jalankan prediksi", type="primary", width="stretch"):
                image_ids = dataset_image_ids(42, batas)
                kunci_param = tuple(sorted(vars(base_params).items()))
                kunci_cache = (kunci_param, batas)
                if st.session_state.get("eval_cache_key") != kunci_cache:
                    with st.spinner(f"Memproses {batas} citra dari dataset…"):
                        try:
                            predictions = predictions_from_dataset(
                                image_ids, base_params, limit=batas
                            )
                            st.session_state["eval_predictions"] = predictions
                            st.session_state["eval_cache_key"] = kunci_cache
                        except Exception as exc:
                            render_failure_card(
                                "Prediksi gagal dijalankan: "
                                f"{exc}. Cek koneksi Kaggle atau unggah CSV prediksi."
                            )
                else:
                    predictions = dict(st.session_state.get("eval_predictions", {}))
            else:
                predictions = dict(st.session_state.get("eval_predictions", {}))
                if predictions:
                    st.caption(
                        "Memakai prediksi dari sesi ini. Geser parameter lalu "
                        "jalankan ulang untuk membandingkan."
                    )
                else:
                    render_empty_state(
                        "Belum ada prediksi",
                        "Tekan Jalankan prediksi setelah acuan siap.",
                    )

    # --------------------------------------------------- 3. Hasil
    with tab_hasil:
        hasil_tersimpan = st.session_state.get("eval_result")
        status_chip(
            "Metrik tersedia untuk baris uji yang sama dengan prediksi."
            if hasil_tersimpan is not None
            else "Isi acuan dan prediksi, lalu lihat metrik di sini.",
            hasil_tersimpan is not None,
        )
        predicted_ids = {str(key) for key in predictions}
        referensi_dipakai = (
            edited[edited["image_id"].astype(str).isin(predicted_ids)].copy()
            if edited is not None
            else pd.DataFrame()
        )
        if not predictions or referensi_dipakai.empty:
            render_empty_state(
                "Metrik belum tersedia",
                "Teks acuan dan prediksi harus menyentuh citra yang sama. Isi "
                "salah satu baris pada tab 1 Acuan, lalu jalankan prediksi.",
            )
        else:
            try:
                hasil = evaluate(
                    referensi_dipakai,
                    predictions,
                    iou_threshold=0.5,
                    seed=42,
                    test_fraction=0.2,
                )
                st.session_state["eval_result"] = hasil
            except ValueError as exc:
                render_failure_card(f"Evaluasi gagal: {exc}")
                hasil = None

            if hasil is not None:
                if filled == 0:
                    render_info_card(
                        "Akurasi plat 0%",
                        "Belum ada teks acuan yang diisi, jadi tidak ada plat "
                        "yang bisa dihitung benar. Isi kolom Teks plat acuan "
                        "pada tab 1 Acuan, lalu metrik akan diperbarui.",
                    )
                st.caption(
                    f"Split seed 42 pada {len(referensi_dipakai)} baris acuan "
                    f"yang diprediksi: {len(hasil.train_ids)} baris latih, "
                    f"{hasil.n} baris uji. Angka di bawah hanya berlaku untuk "
                    "baris uji ini."
                )
                render_stat_grid(
                    [
                        (
                            "IoU rata-rata",
                            f"{hasil.localization['iou_rata_rata']:.2f}",
                            f"ambang {hasil.localization['ambang_iou']:.1f}",
                            hasil.localization["iou_rata_rata"] >= 0.5,
                        ),
                        (
                            "Akurasi plat",
                            f"{hasil.plate_metrics['akurasi_plat'] * 100:.0f}%",
                            "teks sama persis",
                            hasil.plate_metrics["akurasi_plat"] >= 0.5,
                        ),
                        (
                            "Akurasi karakter",
                            f"{hasil.plate_metrics['akurasi_karakter'] * 100:.0f}%",
                            f"{hasil.char_metrics['benar']}/{hasil.char_metrics['total']} karakter",
                            hasil.plate_metrics["akurasi_karakter"] >= 0.5,
                        ),
                    ]
                )
                render_summary_card(
                    [
                        (
                            "IoU rata-rata",
                            hasil.localization["iou_rata_rata"],
                            f"{hasil.localization['iou_rata_rata']:.2f}",
                        ),
                        (
                            "Akurasi plat",
                            hasil.plate_metrics["akurasi_plat"] * 100,
                            f"{hasil.plate_metrics['akurasi_plat'] * 100:.0f}%",
                        ),
                        (
                            "Presisi",
                            hasil.localization["presisi"] * 100,
                            f"{hasil.localization['presisi'] * 100:.0f}%",
                        ),
                    ],
                    note="Ringkasan ini memakai hasil evaluasi terakhir pada baris uji.",
                )
                for amaran in hasil.warnings:
                    render_failure_card(terjemah_pesan(amaran))

                kiri, kanan = st.columns(2)
                with kiri:
                    st.markdown("**Lokalisasi**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Metrik": "Presisi",
                                    "Nilai": f"{hasil.localization['presisi'] * 100:.0f}%",
                                },
                                {
                                    "Metrik": "Daya ingat",
                                    "Nilai": f"{hasil.localization['recall'] * 100:.0f}%",
                                },
                                {
                                    "Metrik": "F1",
                                    "Nilai": f"{hasil.localization['f1'] * 100:.0f}%",
                                },
                                {
                                    "Metrik": "Tingkat deteksi",
                                    "Nilai": f"{hasil.localization['deteksi'] * 100:.0f}%",
                                },
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                with kanan:
                    st.markdown("**Matriks kebingungan teks plat**")
                    st.dataframe(hasil.confusion, width="stretch", height=220)

                with st.expander("Skenario siang/malam, tegak/miring, jauh, dll."):
                    st.dataframe(hasil.scenarios, hide_index=True, width="stretch")
                with st.expander("Ringkasan per kondisi terisi"):
                    if hasil.per_condition.empty:
                        st.caption("Belum ada kolom kondisi yang diisi.")
                    else:
                        st.dataframe(
                            hasil.per_condition, hide_index=True, width="stretch"
                        )
                with st.expander("Waktu rata-rata per tahap"):
                    st.dataframe(hasil.timings, hide_index=True, width="stretch")

    # --------------------------------------- 4. Bandingkan varian
    with tab_banding:
        status_chip(
            "Empat varian pengolahan dibandingkan pada citra yang sama.",
            bool(st.session_state.get("eval_comparison_rows")),
        )
        st.caption(
            "Varian: CLAHE aktif/nonaktif dan binerisasi Otsu/adaptif. "
            "Kolom kondisi acuan dikosongkan agar label varian yang dipakai."
        )
        batas_banding = st.slider(
            "Citra per varian", 4, 12, 6, 2, key="eval_compare_limit"
        )
        if st.button("Bandingkan varian", type="primary", width="stretch"):
            varian = [
                ("CLAHE aktif", "clahe", True, "auto"),
                ("Tanpa CLAHE", "tanpa_clahe", False, "auto"),
                ("Otsu", "otsu", True, "otsu"),
                ("Adaptif", "adaptif", True, "adaptive"),
            ]
            ids_banding = dataset_image_ids(42, batas_banding)
            acuan_netral = (
                edited[
                    edited["image_id"].astype(str).isin(set(ids_banding))
                ].copy()
                if edited is not None
                else pd.DataFrame()
            )
            acuan_netral["condition"] = ""
            baris_banding: list[dict[str, object]] = []
            try:
                for tampilan, label, pakai_clahe, biner in varian:
                    param_varian = PipelineParams(
                        **{
                            **vars(base_params),
                            "use_clahe": pakai_clahe,
                            "binarization": biner,
                        }
                    )
                    with st.spinner(f"Menjalankan {tampilan}…"):
                        prediksi_varian = predictions_from_dataset(
                            ids_banding,
                            param_varian,
                            conditions={name: label for name in ids_banding},
                            limit=batas_banding,
                        )
                    hasil_varian = evaluate(
                        acuan_netral,
                        prediksi_varian,
                        iou_threshold=0.5,
                        seed=42,
                        test_fraction=0.2,
                    )
                    nilai_waktu = [
                        float(item.get("timings_ms", {}).get("total", 0.0))
                        for items in prediksi_varian.values()
                        for item in items[:1]
                    ]
                    baris_banding.append(
                        {
                            "Varian": tampilan,
                            "IoU (%)": round(
                                hasil_varian.localization["iou_rata_rata"] * 100, 1
                            ),
                            "Akurasi plat (%)": round(
                                hasil_varian.plate_metrics["akurasi_plat"] * 100, 1
                            ),
                            "Akurasi karakter (%)": round(
                                hasil_varian.plate_metrics["akurasi_karakter"] * 100, 1
                            ),
                            "Waktu total (ms)": round(
                                sum(nilai_waktu) / len(nilai_waktu)
                                if nilai_waktu
                                else 0.0,
                                1,
                            ),
                        }
                    )
                st.session_state["eval_comparison_rows"] = baris_banding
            except Exception as exc:
                render_failure_card(
                    f"Perbandingan gagal: {exc}. Cek koneksi Kaggle atau "
                    "unggah CSV prediksi."
                )

        baris_banding = list(st.session_state.get("eval_comparison_rows", []))
        if baris_banding:
            tabel = pd.DataFrame(baris_banding)
            st.dataframe(tabel, hide_index=True, width="stretch")
            kolom_grafik = [
                kolom
                for kolom in (
                    "Akurasi plat (%)",
                    "Akurasi karakter (%)",
                    "IoU (%)",
                )
                if kolom in tabel.columns
            ]
            if kolom_grafik:
                st.caption("Perbandingan metrik antarvarian (persen).")
                st.bar_chart(
                    tabel.set_index("Varian")[kolom_grafik],
                    width="stretch",
                )
        else:
            render_empty_state(
                "Belum ada perbandingan",
                "Tekan Bandingkan varian setelah contoh dataset tersedia.",
            )


mark_intro_done()
