"""Halaman Dataset: sumber, struktur, lisensi, dan galeri contoh."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.dataset import DATASET_HANDLE, dataset_status, gallery_items, get_image_bytes
from ui.components import (
    bootstrap,
    mark_intro_done,
    page_header,
    render_empty_state,
    render_failure_card,
    render_performance_panel,
    render_stat_grid,
)

bootstrap("Dataset · Plat Nomor")

REFERENCE_CSV = Path("docs/acuan_ocr.csv")

main, perf = st.columns([1.73, 1], gap="large")

with main:
    page_header(
        "Dataset",
        "Sumber data publik yang dipakai sebagai contoh, bukan sebagai teks acuan otomatis.",
    )

    status = dataset_status()
    if status["available"]:
        st.caption("Status koneksi: **terhubung** ke Kaggle.")
    else:
        render_failure_card(
            status["message"]
            + " Informasi dataset tetap ditampilkan; galeri dinonaktifkan sementara."
        )

    render_stat_grid(
        [
            ("Citra deteksi", "1000", "800 / 100 / 100", True),
            ("Crop pengenalan", "1906", "1530 / 179 / 197", True),
            ("Galeri contoh", "24", "seed 42, cache 1 jam", True),
        ]
    )

    st.subheader("Sumber")
    st.markdown(
        f"- Nama: **Indonesian License Plate Dataset**\n"
        f"- Handle Kaggle: `{DATASET_HANDLE}`\n"
        "- Pembuat: Juan Thomas Wijaya\n"
        "- Versi: 1, diperbarui 2024-12-18\n"
        "- Lisensi menurut metadata: **Unknown**\n"
        "- Ukuran penuh ≈ 2,07 GB; **tidak diunduh seluruhnya**\n"
        "- Halaman: https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset"
    )
    st.caption(
        "Lisensi tidak dikarang. Karena metadata tidak mencantumkan nama/URL lisensi, "
        "dataset diperlakukan hanya untuk keperluan tugas akademik dan kredit ditulis di README."
    )

    st.subheader("Struktur yang dipakai")
    st.dataframe(
        pd.DataFrame(
            [
                {"Folder": "images/test", "Jumlah": 100, "Kegunaan": "Contoh deteksi"},
                {"Folder": "images/train", "Jumlah": 800, "Kegunaan": "Tidak dipakai"},
                {"Folder": "images/val", "Jumlah": 100, "Kegunaan": "Tidak dipakai"},
                {
                    "Folder": "labelswithLP/test",
                    "Jumlah": 100,
                    "Kegunaan": "Kotak acuan IoU; teks tidak dipakai",
                },
                {
                    "Folder": "Recognition Dataset/images/test",
                    "Jumlah": 197,
                    "Kegunaan": "Referensi kelas karakter",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Format label")
    st.code(
        "0 0.235450 0.771991 0.074074 0.022156\n"
        "# class x_center y_center width height (ternormalisasi)",
        language="text",
    )
    st.caption(
        "File `labelswithLP` menambahkan teks di kolom keenam, misalnya `B9140BCD`. "
        "Kolom itu sengaja tidak dipakai sebagai acuan OCR; teks acuan diketik manual."
    )

    st.subheader("Acuan OCR")
    if REFERENCE_CSV.exists():
        st.caption(
            "Template `docs/acuan_ocr.csv` memuat 197 kotak dari 100 citra dengan seed 42. "
            "Kolom `text` dan `condition` masih kosong untuk diisi manual."
        )
        st.download_button(
            "Unduh template acuan",
            data=REFERENCE_CSV.read_bytes(),
            file_name="acuan_ocr.csv",
            mime="text/csv",
        )
    else:
        render_empty_state("Template tidak ditemukan", "Jalankan generator template dari modul core.dataset.")

    st.subheader("Galeri contoh")
    if not status["available"]:
        render_empty_state("Galeri nonaktif", "Koneksi dataset gagal. Halaman Deteksi tetap menerima unggahan lokal.")
    else:
        with st.expander("Lihat 24 contoh dari split test", expanded=False):
            items = gallery_items(42, 24)
            columns = st.columns(6)
            for index, item in enumerate(items):
                with columns[index % 6]:
                    try:
                        st.image(
                            get_image_bytes(item["image_path"]),
                            caption=item["label"],
                            width="stretch",
                        )
                    except Exception as exc:
                        st.caption(f"{item['label']}: gagal dimuat ({exc})")

    st.caption(
        "Setiap file diambil satu per satu dengan `kagglehub.dataset_download(handle, path=...)`. "
        "Kredensial, bila diperlukan, hanya dibaca dari `st.secrets`."
    )

with perf:
    render_performance_panel(
        [
            (
                "Koneksi Kaggle",
                100.0 if status["available"] else 0.0,
                "Tersedia" if status["available"] else "Gagal",
            ),
            ("Galeri", 100.0 if status["available"] else 0.0, "24 citra, seed 42"),
            ("Lisensi", 0.0, "Unknown"),
        ],
        note="Panel ini merangkum status sumber data, bukan hasil akurasi model.",
    )

mark_intro_done()
