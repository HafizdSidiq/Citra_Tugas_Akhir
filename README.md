# Pengenalan Plat Nomor Kendaraan Indonesia

Aplikasi Streamlit untuk membaca plat nomor Indonesia dengan **pengolahan citra klasik** yang transparan: setiap tahap disimpan, ditampilkan, dan diukur waktunya. Tidak memakai PyTorch atau TensorFlow. Detektor YOLO ONNX **opsional** dan hanya dipakai bila file `.onnx` tersedia di repositori; saat ini tidak ada file tersebut.

## Menjalankan aplikasi

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`.

### Tesseract OCR

- **Windows**: pasang [UB Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). `core/ocr.py` otomatis mencari `tesseract.exe` di `PATH`, `C:\Program Files\Tesseract-OCR`, dan `%LOCALAPPDATA%\Programs\Tesseract-OCR`. Jika memakai installer Windows, `TESSDATA_PREFIX` diarahkan ke folder `tessdata`-nya otomatis.
- **Linux / Streamlit Cloud**: paket sistem `tesseract-ocr` dipasang otomatis dari `packages.txt`.
- Konfigurasi OCR: `--oem 3 --psm 7`, whitelist `A-Z0-9`, validasi pola `^[A-Z]{1,2}\d{1,5}[A-Z]{1,3}$`.

### Dataset Kaggle

Dataset: **Indonesian License Plate Dataset** — `juanthomaswijaya/indonesian-license-plate-dataset` ([halaman dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)). Metadata mencantumkan lisensi **Unknown**; proyek memakai data hanya untuk keperluan tugas akademik dan mencantumkan kredit di sini.

Aplikasi **tidak mengunduh seluruh arsip ~2,07 GB**. File diambil satu per satu dengan `kagglehub.dataset_download(handle, path="<file>")`, maksimal 24 citra galeri dengan seed 42, cache `st.cache_data(ttl=3600, max_entries=50)`. Kredensial Kaggle, bila diperlukan, hanya dibaca dari `st.secrets`:

```toml
[kaggle]
username = "..."
key = "..."
```

Unggahan pengguna diproses di memori dan tidak ditulis ke disk. Rincian pemeriksaan dataset ada di `docs/dataset_info.md`.

## Struktur

```text
app.py                  Router st.navigation + header pil melayang
pages/
  0_Beranda.py          Hero simulasi 8 tahap, marquee, kartu bento
  1_Baca_Plat.py        Pilih foto → atur → hasil (satu halaman)
  2_Evaluasi.py         Wizard 4 tab: acuan, prediksi, hasil, varian
  3_Dataset.py          Sumber, lisensi, struktur, galeri malas
  4_Uji_Widget.py       Halaman internal tersembunyi untuk uji kontras
core/
  pipeline.py           Delapan tahap pengolahan citra
  ocr.py                Tesseract, whitelist, validasi plat
  dataset.py            Akses file Kaggle satu per satu, galeri seed 42
  metrics.py            IoU, precision/recall, akurasi, split tanpa kebocoran
  evaluation.py         Prediksi dari CSV atau pipeline untuk evaluasi
  tests/                23 tes pytest
ui/
  theme.css             Token Lukis, Unbounded/Figtree, kartu bulat, mode gelap
  components.py         Komponen tampilan bersama + panggung iframe
docs/
  referensi/lukis.html  Satu-satunya referensi gaya antarmuka
  acuan_ocr.csv         Template 197 kotak dari 100 citra (teks masih kosong)
  dataset_info.md       Metadata dan verifikasi dataset
  screenshots/          Tangkapan layar terang dan gelap per halaman
requirements.txt
packages.txt
```

## Tampilan

Gaya antarmuka mengikuti `docs/referensi/lukis.html`: ungu `#6d3cff`, tosca `#00c4cc`, kuning `#ffd23f`, pink `#ff6b9d`, tinta `#14112b`, dan latar `#fbf9ff`. Judul memakai Unbounded, isi memakai Figtree. Kartu sangat bulat (26–30 px) bergantian antara putih, ungu, kuning, dan tosca tanpa garis dekoratif. Navigasi adalah pil melayang dengan logo berputar dan penanda menu aktif; toolbar bawaan Streamlit disembunyikan.

Semua teks antarmuka berbahasa Indonesia dan menyebut aksinya, misalnya **Baca plat**. Tombol ungu naik 3 px dengan bayangan kuning dan sedikit miring saat dihover, lalu mengecil saat ditekan. Fokus keyboard memakai garis pink. `prefers-reduced-motion` mematikan marquee, logo, garis pindai, dan animasi masuk; `prefers-color-scheme: dark` mengganti token warna tanpa mengorbankan keterbacaan. JavaScript dan SVG hanya disuntik lewat `st.iframe`, bukan `st.markdown`.

Panel ringkasan permanen sudah dihapus. Kartu ringkasan hanya muncul pada halaman dengan hasil nyata. Galeri dataset dan contoh Kaggle baru dimuat setelah tombol ditekan; pemrosesan OCR hanya berjalan setelah **Baca plat**. Halaman uji widget dibuka lewat `/uji-widget` untuk memeriksa kontras seluruh widget.

## Alur pengolahan

1. Resize maksimum 960 px.
2. Grayscale → CLAHE → filter bilateral (CLAHE bisa dimatikan untuk perbandingan).
3. Canny → closing morfologi.
4. Kontur persegi panjang dengan filter rasio `1,6–3,8` dan luas minimum.
5. Koreksi perspektif → binerisasi Otsu atau adaptif.
6. Segmentasi karakter kiri-kanan.
7. OCR Tesseract pada baris hasil segmentasi **dan** pada plat utuh; keyakinan tertinggi dipakai.
8. Kotak anotasi, teks, keyakinan, dan waktu per tahap ditampilkan.

## Evaluasi

- Buka `docs/acuan_ocr.csv` di halaman **Evaluasi**, ketik teks plat acuan dan isi kolom `condition` (siang/malam/bayangan, tegak/miring/jauh, mobil/motor, clahe/tanpa_clahe, otsu/adaptif), lalu unduh CSV terisi.
- Sediakan prediksi lewat tombol **Jalankan prediksi** atau unggah CSV prediksi dengan kolom `image_id,text,box,confidence`.
- Metrik dihitung dengan IoU ≥ 0,5: precision, recall, F1, akurasi plat, akurasi per karakter, confusion matrix, ringkasan skenario, dan waktu rata-rata per tahap.
- Split memakai seed 42 dengan `image_id` sebagai unit, sehingga tidak ada kebocoran antar-split. Angka hanya berlaku untuk baris uji yang dihitung; tidak ada klaim di luar data uji.
- Bagian **Perbandingan varian** menjalankan CLAHE aktif/tanpa dan Otsu/adaptif pada citra yang sama.

## Pengujian

```bash
python -m pytest core/tests -q
python -m compileall -q app.py core ui pages
```

## Batasan yang diketahui

- Teks acuan OCR masih kosong di template; akurasi plat akan bernilai 0% sampai teks diketik manual. Ini disengaja agar acuan tidak diambil dari kolom ke-6 label dataset.
- Pipeline klasik dapat menghasilkan beberapa kandidat per foto (dataset memang memuat lebih dari satu plat). Prediksi banyak kotak dipasangkan sekali pakai secara greedy pada metrik.
- Waktu pemrosesan bergantung mesin dan cache Kaggle; angka di dokumentasi bukan patokan universal.
- Tidak ada model deteksi deep learning yang disertakan.

## Screenshot

Tangkapan layar tiap halaman ada di akar `docs/screenshots/` (tema terang) serta di `docs/screenshots/light/` dan `docs/screenshots/dark/`. Animasi dinonaktifkan saat pengambilan dengan `prefers-reduced-motion`. Tangkapan lama dari desain sebelumnya sudah dihapus.
