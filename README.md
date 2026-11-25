# Plat Scanner

**Plat Scanner** — Pengenalan Plat Nomor Kendaraan Indonesia dengan Pengolahan Citra Klasik dan Tesseract OCR. Aplikasi Streamlit untuk membaca plat nomor Indonesia dengan **pengolahan citra klasik** yang transparan: setiap tahap disimpan, ditampilkan, dan diukur waktunya. Tidak memakai PyTorch atau TensorFlow. Detektor YOLO ONNX **opsional** dan hanya dipakai bila file `.onnx` tersedia di repositori; saat ini tidak ada file tersebut.

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
- Konfigurasi OCR: `--oem 3 --psm 7` dan `--psm 8` dibandingkan, whitelist `A-Z0-9`, validasi pola `^[A-Z]{1,2}\d{1,5}[A-Z]{1,3}$`, lalu koreksi minimal `0/O`, `1/I`, `8/B`, `5/S`, `2/Z`, `6/G` sesuai posisi huruf/angka.

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
app.py                  Router st.navigation + logo statis dan favicon di header pil
pages/
  0_Beranda.py          Identitas, logo animasi, jendela pemindai, dan kartu bento
  1_Baca_Plat.py        Pilih foto → atur → hasil (satu halaman)
  2_Evaluasi.py         Wizard 4 tab: acuan, prediksi, hasil, varian
  3_Dataset.py          Sumber, lisensi, struktur, galeri malas
core/
  pipeline.py           Sembilan tahap pengolahan citra + penilaian kandidat
  ocr.py                Tesseract psm 7/8, whitelist, koreksi posisi plat
  benchmark.py          Patok ukur 60 citra nyata, hasil di docs/benchmark/
  dataset.py            Akses file Kaggle satu per satu, galeri seed 42
  metrics.py            IoU, precision/recall, akurasi, split tanpa kebocoran
  evaluation.py         Prediksi dari CSV atau pipeline untuk evaluasi
  tests/                34 tes pytest
ui/
  theme.css             Token Lukis, Unbounded/Figtree, kartu bulat, tema terang
  components.py         Komponen tampilan bersama + jendela pemindai dan logo
assets/
  logo/                 Logo SCAN statis, animasi, dan favicon
docs/
  referensi/lukis.html  Satu-satunya referensi gaya antarmuka
  acuan_ocr.csv         Template 197 kotak dari 100 citra (teks masih kosong)
  dataset_info.md       Metadata dan verifikasi dataset
  screenshots/          Tangkapan layar tema terang per halaman
requirements.txt
packages.txt
```

## Tampilan

Gaya antarmuka mengikuti `docs/referensi/lukis.html`: ungu `#6d3cff`, tosca `#00c4cc`, kuning `#ffd23f`, pink `#ff6b9d`, tinta `#14112b`, dan latar `#fbf9ff`. Judul memakai Unbounded 800, isi memakai Figtree. Kartu sangat bulat (26–30 px) bergantian antara putih, ungu, kuning, dan tosca tanpa garis dekoratif. Navigasi adalah pil melayang dengan logo SCAN statis berdampingan dengan wordmark **SCAN**; favicon memakai `assets/logo/favicon.png`. Logo animasi pemindai tampil sekali di hero Beranda. Hero memakai judul kapital **PLAT SCANNER**, tagline *Plat terbaca, langkahnya terlihat.*, tiga pil teknologi (OpenCV, Tesseract, Streamlit), dan **jendela pemindai** gelap berisi plat contoh `B1234ABC`, kotak deteksi kuning, siku bidik tosca, garis pindai pink, serta delapan tahap sebagai pil kecil. Hanya jendela pemindai yang gelap; seluruh aplikasi tetap bertema terang. Toolbar bawaan Streamlit disembunyikan.

Aplikasi memakai tema terang saja agar tidak bertabrakan dengan preferensi gelap sistem operasi. Semua teks antarmuka berbahasa Indonesia dan menyebut aksinya, misalnya **Baca plat**. Tombol ungu naik 3 px dengan bayangan kuning dan sedikit miring saat dihover, lalu mengecil saat ditekan. Fokus keyboard memakai garis pink. `prefers-reduced-motion` mematikan garis pindai dan animasi masuk. JavaScript dan SVG hanya disuntik lewat `st.iframe`, bukan `st.markdown`.

Panel ringkasan permanen sudah dihapus. Kartu ringkasan hanya muncul pada halaman dengan hasil nyata. Galeri dataset dan contoh Kaggle baru dimuat setelah tombol ditekan; pemrosesan OCR hanya berjalan setelah **Baca plat**.

## Alur pengolahan

1. Deteksi pada citra maksimum 960 px; faktor skala disimpan.
2. Grayscale → CLAHE → filter bilateral (CLAHE bisa dimatikan untuk perbandingan).
3. Dua ambang Canny (70/210 dan turunannya) digabung **tanpa closing**.
4. Kontur disaring memakai rasio `1,4–3,8`, luas `0,03%–25%` citra, keterisian ≥ 0,6, serta penolakan tepi lebar penuh; duplikat dihapus dengan IoU.
5. Quad diskalakan ke citra asli; koreksi perspektif memakai tinggi adaptif 120–200 px.
6. Crop 3% pinggiran dan 25% bawah sebelum binerisasi Otsu/adaptif.
7. Segmentasi karakter kiri-kanan, lalu OCR Tesseract psm 7/8 pada baris rakitan **dan** plat utuh.
8. Teks dirapikan berdasarkan posisi huruf/angka, lalu kandidat diberi skor pola + keyakinan + geometri.
9. Hanya kandidat dengan skor ≥ ambang yang ditampilkan; anotasi memuat utama dan maksimal satu alternatif terdekat.

## Evaluasi

- Buka `docs/acuan_ocr.csv` di halaman **Evaluasi**, ketik teks plat acuan dan isi kolom `condition` (siang/malam/bayangan, tegak/miring/jauh, mobil/motor, clahe/tanpa_clahe, otsu/adaptif), lalu unduh CSV terisi.
- Sediakan prediksi lewat tombol **Jalankan prediksi** atau unggah CSV prediksi dengan kolom `image_id,text,box,confidence`.
- Metrik dihitung dengan IoU ≥ 0,5: presisi, daya ingat, F1, akurasi plat, akurasi per karakter, matriks kebingungan, ringkasan skenario, dan waktu rata-rata per tahap.
- Split memakai seed 42 dengan `image_id` sebagai unit, sehingga tidak ada kebocoran antar-split. Angka hanya berlaku untuk baris uji yang dihitung; tidak ada klaim di luar data uji.
- Bagian **Perbandingan varian** menjalankan CLAHE aktif/tanpa dan Otsu/adaptif pada citra yang sama.

## Patok ukur lokalisasi dan OCR

```bash
python -m core.benchmark sebelum 60
python -m core.benchmark sesudah 60
```

Perintah memakai 60 citra test Kaggle yang sama dengan `docs/acuan_ocr.csv`, mengukur recall kotak acuan pada IoU ≥ 0,5, hit kandidat utama, rata-rata kotak per citra, akurasi plat, dan akurasi per karakter. Hasil disimpan di `docs/benchmark/<label>.json` beserta parameter yang dipakai. Kolom `text` CSV tidak diubah; untuk metrik teks, acuan diambil dari kolom ke-6 label YOLO yang sudah tersedia di cache dataset.

## Pengujian

```bash
python -m pytest core/tests -q
python -m compileall -q app.py core ui pages
```

## Batasan yang diketahui

- Patok ukur 60 citra test (`docs/benchmark/sebelum.json` → `docs/benchmark/sesudah.json`): recall lokalisasi **0,85% → 22,03%**, hit top-1 **1,67% → 25,00%**, rata-rata kotak final per citra **5,833 → 0,483**. Akurasi plat **0,85%** dan akurasi per karakter **51,52%** dihitung dari 118 teks acuan; angka ini masih rendah dan tidak diklaim sebagai pengenalan plat yang andal.
- Kolom `text` di `docs/acuan_ocr.csv` sengaja tetap kosong. Metrik teks pada patok ukur memakai kolom ke-6 label YOLO yang sudah ada di cache dataset, bukan acuan manual; evaluasi di halaman aplikasi tetap memakai teks yang diketik pengguna.
- Ambang skor 55 menekan kotak palsu, tetapi mengorbankan sebagian hit top-1 (dari 0,28 menjadi 0,25 pada 60 citra). Kandidat dengan teks salah masih mungkin lolos bila polanya valid.
- Pipeline klasik dapat menghasilkan beberapa kandidat per foto (dataset memang memuat lebih dari satu plat). Prediksi banyak kotak dipasangkan sekali pakai secara greedy pada metrik.
- Waktu pemrosesan bergantung mesin dan cache Kaggle; angka di dokumentasi bukan patokan universal.
- Tidak ada model deteksi deep learning yang disertakan.

## Screenshot

Tangkapan layar tiap halaman ada di akar `docs/screenshots/` dan `docs/screenshots/light/` dengan tema terang. `docs/screenshots/beranda_preferensi_gelap.png` menunjukkan tampilan tetap terang saat browser meminta mode gelap. Animasi dinonaktifkan saat pengambilan dengan `prefers-reduced-motion`. Folder `docs/screenshots/dark/` sengaja tidak dipakai karena aplikasi tetap terang saja, termasuk logo statis dan animasinya. Contoh hasil OCR pada lima foto berbeda ada di `docs/screenshots/ocr/` beserta `laporan.json` (teks yang terbaca dan error browser, kalau ada). Label siang/malam/miring/motor/mobil adalah deskripsi sampel dataset, bukan anotasi resmi; teks hasil sengaja dilaporkan apa adanya, termasuk yang salah.
