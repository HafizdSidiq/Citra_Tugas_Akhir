"""Halaman Dataset: sumber, lisensi, struktur, dan galeri contoh."""

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
    render_stat_grid,
)

bootstrap()

REFERENCE_CSV = Path("docs/acuan_ocr.csv")

with st.container(border=True, key="kartu_dataset"):
    page_header(
        "Dataset",
        "Sumber data publik yang dipakai sebagai contoh, bukan sebagai teks acuan otomatis.",
    )
    render_stat_grid(
        [
            ("Citra deteksi", "1000", "800 / 100 / 100", True),
            ("Crop pengenalan", "1906", "1530 / 179 / 197", True),
            ("Galeri contoh", "24", "seed 42, cache 1 jam", True),
        ]
    )

with st.container(border=True, key="kartu_sumber"):
    st.markdown("**Sumber dataset**")
    st.markdown(
        f"- Nama: **Indonesian License Plate Dataset**\n"
        f"- Handle Kaggle: `{DATASET_HANDLE}`\n"
        "- Pembuat: Juan Thomas Wijaya\n"
        "- Versi: 1, diperbarui 2024-12-18\n"
        "- Ukuran penuh ≈ 2,07 GB; **tidak diunduh seluruhnya**\n"
        "- Halaman: https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset"
    )

with st.container(border=True, key="kartu_lisensi"):
    st.markdown("**Lisensi dan batasan**")
    st.markdown("Lisensi menurut metadata: **Unknown**.")
    st.caption(
        "Nama lisensi tidak dikarang. Karena metadata tidak mencantumkannya, "
        "dataset hanya dipakai untuk keperluan tugas akademik dan kredit ditulis di README. "
        "Nilai ini sengaja tidak diubah menjadi persentase."
    )

with st.container(border=True, key="kartu_struktur"):
    st.markdown("**Struktur yang dipakai**")
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
    st.caption(
        "Berkas `labelswithLP` menambahkan teks di kolom keenam, misalnya `B9140BCD`. "
        "Kolom itu sengaja tidak dipakai sebagai acuan OCR; teks acuan diketik manual."
    )
    st.code(
        "0 0.235450 0.771991 0.074074 0.022156\n"
        "# class x_center y_center width height (ternormalisasi)",
        language="text",
    )

with st.container(border=True, key="kartu_acuan"):
    st.markdown("**Template acuan OCR**")
    if REFERENCE_CSV.exists():
        st.caption(
            "`docs/acuan_ocr.csv` memuat 197 kotak dari 100 citra dengan seed 42. "
            "Kolom `text` dan `condition` masih kosong untuk diisi manual."
        )
        st.download_button(
            "Unduh template acuan",
            data=REFERENCE_CSV.read_bytes(),
            file_name="acuan_ocr.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        render_empty_state(
            "Template tidak ditemukan",
            "Jalankan generator template dari modul core.dataset.",
        )

with st.container(border=True, key="kartu_galeri"):
    st.markdown("**Galeri contoh**")
    st.caption(
        "Contoh diambil satu per satu hanya setelah tombol ditekan. Rerun tanpa "
        "interaksi tidak mengunduh berkas baru."
    )
    if st.button("Periksa koneksi dan muat 24 contoh", key="dataset_muat"):
        st.session_state["dataset_status"] = dataset_status()
        st.session_state["dataset_galeri_dimuat"] = True

    status = st.session_state.get("dataset_status")
    if status is None:
        render_empty_state(
            "Galeri belum dimuat",
            "Tekan tombol di atas untuk memeriksa koneksi Kaggle dan mengambil contoh.",
        )
    elif not status["available"]:
        render_failure_card(
            status["message"] + " Halaman Baca Plat tetap menerima unggahan lokal."
        )
    else:
        st.caption("Status koneksi: **terhubung** ke Kaggle.")
        with st.expander("Lihat 24 contoh dari split test", expanded=False):
            items = gallery_items(42, 24)
            kolom = st.columns(6)
            for indeks, item in enumerate(items):
                with kolom[indeks % 6]:
                    st.markdown('<div class="gallery-item">', unsafe_allow_html=True)
                    try:
                        st.image(get_image_bytes(item["image_path"]), width="stretch")
                    except Exception as exc:
                        st.caption(f"{item['label']}: gagal dimuat ({exc})")
                    st.caption(item["label"])
                    st.markdown("</div>", unsafe_allow_html=True)

mark_intro_done()
