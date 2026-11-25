"""Halaman internal untuk memeriksa kontras dan perilaku semua widget."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import bootstrap, mark_intro_done, page_header

bootstrap()

with st.container(border=True, key="kartu_widget"):
    page_header(
        "Uji Widget",
        "Halaman internal untuk memeriksa keterbacaan semua widget dalam mode terang, gelap, dan layar sempit.",
    )

    st.subheader("Tombol")
    kolom1, kolom2, kolom3 = st.columns(3)
    with kolom1:
        st.button("Baca plat", type="primary", width="stretch")
    with kolom2:
        st.button("Pilih contoh", width="stretch")
    with kolom3:
        st.download_button(
            "Unduh contoh CSV",
            data="image_id,text\ncontoh.jpg,\n",
            file_name="contoh.csv",
            mime="text/csv",
            width="stretch",
        )

    st.subheader("Input")
    st.file_uploader("Unggah foto uji", type=["jpg", "png"], key="uji_unggah")
    st.radio("Kondisi", ["Siang", "Malam", "Miring"], horizontal=True, key="uji_radio")
    st.selectbox("Binerisasi", ["auto", "otsu", "adaptive"], key="uji_select")
    st.multiselect("Jenis kendaraan", ["Mobil", "Motor", "Truk"], key="uji_multi")
    st.slider("Ambang Canny bawah", 20, 120, 50, key="uji_slider")
    st.checkbox("Gunakan CLAHE", value=True, key="uji_check")
    st.text_input("Teks plat acuan", placeholder="B1234ABC", key="uji_text")
    st.text_area("Catatan", "Tulis catatan singkat di sini.", key="uji_area")
    st.number_input("Jumlah citra", 1, 50, 12, key="uji_number")
    st.date_input("Tanggal uji", key="uji_date")
    st.time_input("Waktu uji", key="uji_time")

    st.subheader("Penampil")
    with st.expander("Parameter pengolahan", expanded=False):
        st.caption("Isi expander ini memakai warna soft dan teks berkontras tinggi.")
        st.progress(0.65, text="Contoh bilah progres")
    tab_tabel, tab_editor, tab_status = st.tabs(["Tabel", "Editor", "Status"])
    with tab_tabel:
        contoh = pd.DataFrame(
            [
                {"image_id": "test001", "Teks": "B1234ABC", "Kondisi": "siang"},
                {"image_id": "test002", "Teks": "", "Kondisi": "malam"},
            ]
        )
        st.dataframe(contoh, hide_index=True, width="stretch")
    with tab_editor:
        st.data_editor(contoh, hide_index=True, width="stretch", key="uji_editor")
    with tab_status:
        with st.status("Pemeriksaan selesai", state="complete", expanded=True):
            st.write("Semua widget tampil pada halaman ini.")

    st.caption("Teks kecil harus tetap terbaca di atas latar terang, gelap, dan kartu berwarna.")

mark_intro_done()
