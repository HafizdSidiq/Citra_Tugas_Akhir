# Informasi Dataset

Diperiksa: 2026-11-25.

## Sumber

- Nama: **Indonesian License Plate Dataset**
- Handle Kaggle: `juanthomaswijaya/indonesian-license-plate-dataset`
- Versi: 1 (`lastUpdated` 2024-12-18)
- Halaman: https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset
- Pembuat: Juan Thomas Wijaya
- Lisensi menurut metadata Kaggle: **Unknown** (tidak ada nama/URL lisensi yang tercantum). Artikel/skripsi belum tersedia; deskripsi menyatakan referensi akan diperbarui. **Asumsi proyek:** dataset dipakai hanya untuk keperluan tugas akademik dan kredit ditulis di README.
- Ukuran total menurut metadata: 2 067 093 167 byte (~2,07 GB). Tidak diunduh seluruhnya.

Cara pemeriksaan: metadata publik Kaggle dan endpoint daftar file (`datasets.DatasetApiService/ListTreeDatasetFiles`), lalu mengambil **beberapa file kecil satu per satu** dengan `kagglehub.dataset_download(handle, path="<file>")`. Daftar lengkap file diperoleh tanpa mengunduh isi dataset.

## Struktur file

Path di bawah relatif terhadap akar dataset. Nama folder memang berulang pada zip aslinya.

```text
Indonesian License Plate Dataset/
  images/
    test/    100 file .jpg
    train/   800 file .jpg
    val/     100 file .jpg
  labels/
    test/    100 file .txt
    train/   800 file .txt
    val/     100 file .txt
  labelswithLP/
    test/    100 file .txt   (sama seperti labels, + teks plat)
    train/   800 file .txt
    val/     100 file .txt

Indonesian License Plate Recognition Dataset/
  classes.names
  train.txt                 (isi: satu baris kosong)
  val.txt                   (isi: satu baris kosong)
  images/
    test/    197 file .jpg
    train/  1530 file .jpg
    val/     179 file .jpg
  labels/
    test/    197 file .txt
    train/  1530 file .txt
    val/     179 file .txt
```

Jumlah sesuai deskripsi pembuat: 1000 citra deteksi dan 1906 plat crop (197 + 1530 + 179 = 1906).

## Format label

### Deteksi (`labels/<split>/<nama>.txt`)

YOLO, satu baris per plat:

```text
0 0.235450 0.771991 0.074074 0.022156
```

Artinya: `class x_center y_center width height`, semua koordinat dinormalisasi terhadap lebar/tinggi citra. Kelas 0 = plat.

### Deteksi + teks (`labelswithLP/<split>/<nama>.txt`)

Format sama, dengan kolom keenam berupa teks plat:

```text
0 0.23545 0.771991 0.074074 0.022156 B9140BCD
```

Contoh lain: `H8518NA`, `K1649GB`, `BG1352AE`. Kolom ini berguna sebagai referensi, tetapi sesuai ketentuan tugas, CSV acuan OCR untuk evaluasi tetap akan diketik manual pada sampel uji.

### Pengenalan karakter (`Recognition Dataset/labels/...`)

YOLO dengan 36 kelas. `classes.names` berurutan: `0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`. Contoh:

```text
4 0.451827 0.399471 0.083056 0.608466
```

## Verifikasi kecil pada data

- Citra deteksi menurut deskripsi: 3024 × 3024 px, diambil dengan ponsel.
- Citra crop pengenalan yang diambil sebagai contoh: **602 × 189 px** (rasio 3,19:1).
- Sampel 40 file `labelswithLP/test` (seed 42) menghasilkan 82 kotak plat.
- Jumlah plat per citra pada sampel: 1 (10 citra), 2 (20), 3 (8), 4 (2). Artinya satu foto bisa berisi beberapa plat.
- Rasio lebar:tinggi kotak plat pada sampel: minimum 1,513; p10 2,227; median 2,800; p90 3,168; maksimum 3,506.
- 78 dari 82 teks cocok dengan pola `^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}$`. Beberapa plat memakai angka `0` di bagian akhir huruf, sehingga whitelist OCR harus tetap `A-Z0-9` dan validasi pola harus dilonggarkan sampai saat OCR.

**Asumsi parameter awal (akan diuji, bukan klaim):** filter kontur memakai rentang rasio lebar ±1,6–3,8 dengan bobot preferensi sekitar 2,4 (motor) dan 3,2 (mobil); luas minimum relative terhadap citra. Nilai final ditentukan dari hasil pengujian pada halaman Evaluasi.

## Rencana pemakaian di aplikasi

- Galeri maksimal 24 citra dari `Indonesian License Plate Dataset/images/test`, dipilih dengan seed tetap 42, diambil satu per satu saat dibutuhkan, cache `st.cache_data(ttl=3600, max_entries=50)`.
- Label YOLO dipakai untuk kotak acuan IoU. Teks acuan OCR tetap berasal dari CSV isian manual (templat disediakan).
- Tidak ada model `.onnx` dalam dataset; detektor pembanding opsional tidak akan diunduh dari luar daftar ini.
- Kredensial Kaggle (jika diperlukan) hanya dibaca dari `st.secrets`; tidak disimpan ke disk oleh aplikasi. Saat pemeriksaan ini, folder `~/.kaggle/kaggle.json` tidak ada, tetapi unduhan file publik tetap berhasil.

## Template acuan OCR

`docs/acuan_ocr.csv` dibuat dari 100 citra `test` dengan seed 42. File ini berisi **197 baris kotak** (satu foto bisa memuat lebih dari satu plat). Kolom `box` diisi dari label YOLO dalam satuan piksel citra asli (3024 × 3024), sedangkan kolom `text` dan `condition` **sengaja kosong** untuk diketik manual. Kolom ke-6 `labelswithLP` tidak dipakai sebagai teks acuan.

Evaluasi memakai split seed 42 tanpa kebocoran: semua baris dengan `image_id` sama masuk split yang sama. Hanya baris yang juga memiliki prediksi yang dihitung; metrik tidak diklaim berlaku untuk citra di luar data uji.
