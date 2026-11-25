# Ringkasan benchmark 60 citra test Kaggle

Sumber: `juanthomaswijaya/indonesian-license-plate-dataset`, split `test`. Kotak acuan: `docs/acuan_ocr.csv` (118 kotak). Teks acuan untuk metrik OCR: kolom ke-6 label YOLO yang sudah ada di cache dataset; `docs/acuan_ocr.csv` tidak diubah.

| Tahap | Recall lokalisasi (IoU ≥ 0,5) | Hit top-1 | Kotak final/citra | Akurasi plat | Akurasi karakter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sebelum (`sebelum.json`) | 0,0085 | 0,0167 | 5,833 | tidak tersedia | tidak tersedia |
| Langkah 1 (`langkah1.json`) | 0,2203 | 0,2500 | 3,833 | tidak tersedia | tidak tersedia |
| Langkah 2 (`langkah2.json`) | 0,2203 | 0,0333 | 0,033 | 0,0000 | 0,8667\* |
| Langkah 3 (`langkah3.json`) | 0,2203 | 0,2500 | 0,483 | 0,0085 | 0,5152 |
| Sesudah (`sesudah.json`) | 0,2203 | 0,2500 | 0,483 | 0,0085 | 0,5152 |

\* Angka langkah 2 hanya berasal dari dua kotak final di seluruh 60 citra, jadi tidak representatif. Angka sesudah memakai 118 teks acuan.

Catatan:

- Langkah 1 mengganti Canny + closing 17×2 dengan dua Canny tanpa closing, menambah filter luas maksimum/keterisian/tepi, dan deduplikasi IoU.
- Langkah 2 menambah skor, ambang, dan pengurutan; penyaringan yang terlalu ketat membuat top-1 turun sementara kotak palsu hilang.
- Langkah 3 memakai crop 25% bawah + 3% samping, psm 7/8, rakitan karakter, dan koreksi posisi huruf/angka. Top-1 kembali ke 0,25 dengan kotak jauh lebih sedikit.
- Langkah 4 hanya mengubah UI/widget; pipeline dan angka tidak berubah dari langkah 3.
- Tuning tambahan pada 50 citra ada di berkas `tuning_*.json` untuk memilih crop, psm, jarak rakitan, dan ambang skor.
- Semua angka hanya berlaku untuk 60 citra yang diproses; tidak ada klaim generalisasi di luar sampel ini.
