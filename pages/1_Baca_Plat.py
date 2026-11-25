"""Baca Plat: tiga kartu berurutan dari pemilihan foto hingga hasil."""

from __future__ import annotations

import hashlib
import time

import pandas as pd
import streamlit as st

from core.dataset import DatasetError, dataset_status, gallery_items, get_image_bytes
from core.pipeline import PipelineParams, process_image
from ui.components import (
    append_history,
    bootstrap,
    mark_intro_done,
    page_header,
    parameter_widgets,
    performance_from_result,
    render_copy_button,
    render_empty_state,
    render_failure_card,
    render_image_card,
    render_plate_headline,
    render_scan_overlay,
    render_summary_card,
)

bootstrap()

PRESET = {
    "Standar": {
        "canny_low": 50,
        "canny_high": 150,
        "min_area": 0.001,
        "aspect_min": 1.6,
        "aspect_max": 3.8,
        "use_clahe": True,
        "binarization": "auto",
    },
    "Foto malam": {
        "canny_low": 30,
        "canny_high": 100,
        "min_area": 0.0005,
        "aspect_min": 1.6,
        "aspect_max": 3.8,
        "use_clahe": True,
        "binarization": "adaptive",
    },
    "Plat miring": {
        "canny_low": 40,
        "canny_high": 120,
        "min_area": 0.0006,
        "aspect_min": 1.4,
        "aspect_max": 4.2,
        "use_clahe": True,
        "binarization": "auto",
    },
}

TAHAPAN_TAMPIL = [
    ("resize", "Perkecil citra", "Citra diperkecil maksimal 960 px dengan rasio tetap."),
    ("grayscale", "Ubah ke abu-abu", "Warna dihapus agar kontras teks lebih mudah diukur."),
    ("enhanced", "Perkuat kontras", "CLAHE dan filter bilateral menaikkan kontras sekaligus menekan noise."),
    ("edges", "Temukan tepi", "Canny menandai tepi, lalu closing menyambung bagian yang putus."),
    ("contours", "Pilih kontur plat", "Kontur persegi panjang disaring memakai rasio dan luas."),
    ("rectified_1", "Luruskan plat", "Sudut plat dikoreksi menjadi persegi panjang tetap."),
    ("otsu_1", "Binerisasi", "Plat diubah ke hitam putih dengan Otsu atau ambang adaptif."),
    ("characters_1", "Segmentasi karakter", "Karakter dipisahkan dari kiri ke kanan sebelum dibaca OCR."),
]


@st.cache_data(show_spinner=False)
def _proses_tersimpan(digest: str, kunci_params: tuple, payload: bytes):
    """Jalankan pipeline sekali per kombinasi citra dan parameter."""
    del digest  # sudah menjadi bagian kunci cache
    return process_image(payload, PipelineParams(**dict(kunci_params)))


def _terapkan_preset() -> None:
    """Salin nilai preset ke widget sebelum widget dibuat."""
    nilai = PRESET[st.session_state["preset_baca"]]
    for kunci, isi in nilai.items():
        st.session_state[kunci] = isi


def _terjemah_pesan(pesan: str) -> str:
    """Samakan istilah luaran core dengan bahasa antarmuka."""
    return str(pesan).replace("miss", "luput")


with st.container(border=True, key="kartu_baca"):
    page_header(
        "Baca Plat",
        "Pilih foto, atur pengolahan, lalu baca platnya. Pemrosesan berjalan "
        "hanya ketika tombol Baca plat ditekan.",
    )

    # -------------------------------------------------- 1. Pilih foto
    st.subheader("1 · Pilih foto")
    tab_unggah, tab_contoh = st.tabs(["Unggah", "Contoh dataset"])

    with tab_unggah:
        unggahan = st.file_uploader(
            "Pilih berkas foto kendaraan (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="unggahan_foto",
        )
        if unggahan is not None:
            st.session_state["sumber_bytes"] = unggahan.getvalue()
            st.session_state["sumber_nama"] = unggahan.name
            st.session_state["sumber_jenis"] = "Unggah"
        else:
            st.caption(
                "Belum ada foto. Pilih berkas di area ini atau buka tab Contoh dataset."
            )

    with tab_contoh:
        if st.button("Muat 24 contoh dari Kaggle", key="muat_galeri"):
            st.session_state["galeri_dimuat"] = True
        if st.session_state.get("galeri_dimuat"):
            status = dataset_status()
            if not status["available"]:
                render_failure_card(
                    status["message"]
                    + " Galeri dinonaktifkan sementara; unggah foto lokal tetap berfungsi."
                )
            else:
                st.caption(
                    "Klik thumbnail untuk memilih contoh. Unduhan terjadi setelah "
                    "tombol ditekan, bukan saat halaman dimuat."
                )
                contoh = gallery_items(42, 24)
                kolom = st.columns(4)
                for indeks, item in enumerate(contoh):
                    with kolom[indeks % 4]:
                        try:
                            data = get_image_bytes(item["image_path"])
                        except DatasetError:
                            st.caption(f"{item['label']}: gagal dimuat")
                            continue
                        st.markdown('<div class="gallery-item">', unsafe_allow_html=True)
                        st.image(data, width="stretch")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if st.button(
                            item["label"],
                            key=f"pilih_{item['label']}",
                            width="stretch",
                        ):
                            st.session_state["contoh_terpilih"] = item
                            st.session_state["sumber_bytes"] = data
                            st.session_state["sumber_nama"] = item["label"] + ".jpg"
                            st.session_state["sumber_jenis"] = "Contoh dataset"
                terpilih = st.session_state.get("contoh_terpilih")
                if terpilih:
                    st.caption(
                        f"Contoh terpilih: **{terpilih['label']}**. "
                        "Lanjut ke langkah 3 lalu tekan Baca plat."
                    )
        else:
            render_empty_state(
                "Galeri belum dimuat",
                "Tombol di atas mengambil 24 contoh hanya saat Anda membuka pilihan ini.",
            )

    sumber_bytes = st.session_state.get("sumber_bytes")
    sumber_nama = str(st.session_state.get("sumber_nama", ""))
    if sumber_bytes is not None:
        st.caption(f"Sumber aktif: **{sumber_nama}**")
        pratinjau, _ruang = st.columns([1, 1])
        with pratinjau:
            st.image(sumber_bytes, caption=sumber_nama, width="stretch")

    tombol_baca = st.button(
        "Baca plat",
        type="primary",
        width="stretch",
        disabled=sumber_bytes is None,
        key="tombol_baca",
    )
    if sumber_bytes is not None:
        st.caption(
            "Tombol ini memproses foto di atas dengan pengaturan yang sedang aktif."
        )

    # ----------------------------------------------------- 2. Atur
    st.subheader("2 · Atur")
    st.selectbox(
        "Preset pengolahan",
        list(PRESET),
        key="preset_baca",
        on_change=_terapkan_preset,
        help="Preset mengisi slider di bawah. Anda masih bisa mengubahnya sendiri.",
    )
    params = parameter_widgets()

    # ---------------------------------------------------- 3. Hasil
    st.subheader("3 · Hasil")
    st.caption(
        "Hasil muncul di sini setelah tombol Baca plat ditekan. Citra dan parameter "
        "yang sama disimpan sementara agar tidak dihitung ulang."
    )
    if tombol_baca:
        if sumber_bytes is None:
            render_failure_card(
                "Belum ada citra yang dipilih. Pilih foto pada Langkah 1 atau "
                "muat contoh dataset, lalu tekan Baca plat kembali."
            )
        else:
            render_scan_overlay()
            progress = st.progress(0.0, text="Menyiapkan citra…")
            kunci_params = tuple(sorted(vars(params).items()))
            digest = hashlib.sha256(sumber_bytes).hexdigest()
            try:
                with st.status("Membaca plat…", expanded=True) as status:
                    hasil = _proses_tersimpan(digest, kunci_params, sumber_bytes)
                    for indeks, (_, judul, _) in enumerate(TAHAPAN_TAMPIL, start=1):
                        status.write(f"{indeks}. {judul}")
                        progress.progress(
                            indeks / len(TAHAPAN_TAMPIL),
                            text=f"Tahap {indeks} dari {len(TAHAPAN_TAMPIL)}: {judul}",
                        )
                        time.sleep(0.03)
                    status.update(label="Pembacaan selesai", state="complete", expanded=False)
                st.session_state["result"] = hasil
                append_history(hasil.text or "?", hasil.confidence)
            except Exception as exc:
                render_failure_card(
                    "Pemrosesan tidak berhasil. Periksa berkas Tesseract dan format "
                    f"citra, lalu coba lagi. Rincian: {exc}"
                )

    result = st.session_state.get("result")
    if result is None:
        render_empty_state(
            "Hasil belum tersedia",
            "Pilih foto pada Langkah 1, lalu tekan tombol Baca plat.",
        )
    else:
        if result.text:
            render_plate_headline(result.text, result.confidence, len(result.plates))
            render_copy_button(result.text)
        else:
            render_failure_card(
                "Teks plat belum terbaca. Coba foto yang lebih dekat dan tajam, "
                "atau sesuaikan parameter pada Langkah 2."
            )

        render_image_card(
            result.annotated_image,
            "Citra dengan kotak plat terdeteksi",
            css_class="result-card",
        )

        for amaran in result.warnings:
            render_failure_card(_terjemah_pesan(amaran))

        if result.plates:
            baris = [
                {
                    "Plat": plate.index,
                    "Teks": plate.text or "—",
                    "Keyakinan (%)": plate.confidence,
                    "Pola valid": "Ya" if plate.valid_pattern else "Tidak",
                    "Binerisasi": plate.threshold_used,
                    "Karakter": len(plate.char_boxes),
                }
                for plate in result.plates
            ]
            st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")

        tab_tahapan, tab_waktu = st.tabs(["Tahapan", "Waktu"])
        with tab_tahapan:
            st.caption("Delapan tahap pengolahan, masing-masing satu kalimat penjelas.")
            tersedia = {
                kunci: gambar
                for kunci, gambar in result.stages.items()
                if not any(kunci.endswith(f"_{i}") for i in range(2, 7))
            }
            for pasangan in range(0, len(TAHAPAN_TAMPIL), 2):
                kiri, kanan = st.columns(2)
                for kolom, (kunci, judul, penjelasan) in zip(
                    (kiri, kanan), TAHAPAN_TAMPIL[pasangan : pasangan + 2]
                ):
                    with kolom:
                        gambar = tersedia.get(kunci)
                        if gambar is None:
                            st.caption(f"{judul}: tahap tidak tersedia pada citra ini.")
                            continue
                        render_image_card(
                            gambar,
                            f"{judul} — {penjelasan}",
                            delay_ms=pasangan * 90,
                        )
        with tab_waktu:
            waktu = pd.DataFrame(
                [
                    {"Tahap": nama, "Waktu (ms)": nilai}
                    for nama, nilai in result.timings_ms.items()
                ]
            )
            st.dataframe(waktu, hide_index=True, width="stretch")

        render_summary_card(
            performance_from_result(result),
            note="Ringkasan ini muncul hanya setelah ada hasil pembacaan.",
        )

mark_intro_done()
