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
app.py                  Beranda + panel performa
pages/
  1_Deteksi.py          Unggah/contoh citra, hasil OCR
  2_Proses.py           Kartu tahap, parameter, waktu per tahap
  3_Evaluasi.py         Acuan OCR, prediksi, metrik, perbandingan varian
  4_Dataset.py          Sumber, struktur, lisensi, galeri
core/
  pipeline.py           Delapan tahap pengolahan citra
  ocr.py                Tesseract, whitelist, validasi plat
  dataset.py            Akses file Kaggle satu per satu, galeri seed 42
  metrics.py            IoU, precision/recall, akurasi, split tanpa kebocoran
  evaluation.py         Prediksi dari CSV atau pipeline untuk evaluasi
  tests/                23 tes pytest
ui/
  theme.css             Tiga zona, warna, tipografi, radius
  motion.css            Animasi CSS murni + prefers-reduced-motion
  components.py         Komponen tampilan bersama
docs/
  acuan_ocr.csv         Template 197 kotak dari 100 citra (teks masih kosong)
  dataset_info.md       Metadata dan verifikasi dataset
  screenshots/          Tangkapan layar tiap halaman
requirements.txt
packages.txt
```

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

Tangkapan layar tiap halaman tersimpan di `docs/screenshots/`. Animasi dinonaktifkan saat pengambilan dengan `prefers-reduced-motion`.
