"""Pembacaan teks plat dengan Tesseract OCR.

Semua konfigurasi OCR bersifat eksplisit agar mudah dijelaskan: OCR memakai
whitelist A-Z0-9 dan page segmentation mode 7 atau 8, lalu teks dirapikan
berdasarkan posisi huruf/angka pola plat Indonesia.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytesseract

# Pola plat Indonesia umum: 1-2 huruf awal, 1-5 angka, 1-3 huruf akhir.
# rentang angka diperlonggar karena sampel dataset mengandung S61310CF.
PLATE_PATTERN = re.compile(r"^[A-Z]{1,2}\d{1,5}[A-Z]{1,3}$")
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Pasangan yang sering tertukar Tesseract pada plat biner.
ANGKA_KE_HURUF = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}
HURUF_KE_ANGKA = {v: k for k, v in ANGKA_KE_HURUF.items()}


def tesseract_config(psm: int = 7) -> str:
    """Buat konfigurasi Tesseract untuk satu mode segmentasi halaman."""
    return (
        f"--oem 3 --psm {int(psm)} -c tessedit_char_whitelist={WHITELIST}"
    )


TESSERACT_CONFIG = tesseract_config(7)


class OcrUnavailableError(RuntimeError):
    """Tesseract tidak bisa dipakai, lengkap dengan saran perbaikan."""


@dataclass(frozen=True)
class OcrOutput:
    """Hasil OCR mentah dan sesudah dirapikan."""

    text: str
    normalized: str
    confidence: float
    valid_pattern: bool


@lru_cache(maxsize=1)
def configure_tesseract() -> str | None:
    """Cari biner Tesseract dan daftarkan ke pytesseract.

    Tidak menulis apa pun ke disk. Lokasi umum Windows dicek sebagai cadangan
    karena installer UB Mannheim tidak selalu menambahkan PATH pada sesi lama.
    """
    exe = shutil.which("tesseract")
    from_path = bool(exe)
    if not exe:
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Tesseract-OCR"
            / "tesseract.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.exe",
            Path("/usr/bin/tesseract"),
            Path("/usr/local/bin/tesseract"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                exe = str(candidate)
                break
    if not exe:
        return None

    pytesseract.pytesseract.tesseract_cmd = exe
    if not from_path:
        # Installer Windows menyimpan traineddata di <root>\tessdata.
        # Tesseract membaca TESSDATA_PREFIX sebagai direktori tessdata itu sendiri.
        tessdata = Path(exe).parent / "tessdata"
        if tessdata.is_dir():
            os.environ["TESSDATA_PREFIX"] = str(tessdata)
    return exe


def normalize_plate_text(text: str) -> str:
    """Buang spasi/pemisah dan samakan huruf ke kapital."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def validate_plate(text: str) -> bool:
    """Periksa apakah teks mengikuti pola huruf-angka-huruf."""
    return bool(PLATE_PATTERN.fullmatch(normalize_plate_text(text)))


def _ubah_per_posisi(karakter: str, tempat: str) -> str | None:
    """Ubah satu karakter agar cocok dengan tempat 'huruf' atau 'angka'."""
    if tempat == "huruf":
        if karakter.isdigit():
            return ANGKA_KE_HURUF.get(karakter)
        return karakter if karakter.isalpha() else None
    if karakter.isalpha():
        return HURUF_KE_ANGKA.get(karakter.upper())
    return karakter if karakter.isdigit() else None


def correct_plate_positions(text: str) -> str:
    """Perbaiki 0/1/8/5/2/6 yang tertukar berdasarkan posisi pola plat.

    Dicoba semua pembagian huruf awal (1-2), angka (1-5), huruf akhir (1-3).
    Pembagian dengan jumlah koreksi paling sedikit yang dipilih. Karakter di
    luar pasangan yang dikenal tidak diubah, sehingga teks tetap utuh bila
    tidak ada pembagian yang valid.
    """
    asli = normalize_plate_text(text)
    n = len(asli)
    if not (3 <= n <= 10):
        return asli
    if validate_plate(asli):
        return asli

    usulan: list[tuple[int, int, int, str]] = []
    for awal in (1, 2):
        for akhir in (1, 2, 3):
            tengah = n - awal - akhir
            if not 1 <= tengah <= 5:
                continue
            bagian = [
                (0, awal, "huruf"),
                (awal, awal + tengah, "angka"),
                (awal + tengah, n, "huruf"),
            ]
            hasil: list[str] = []
            koreksi = 0
            for mulai, selesai, tempat in bagian:
                for karakter in asli[mulai:selesai]:
                    ubah = _ubah_per_posisi(karakter, tempat)
                    if ubah is None:
                        koreksi = -1
                        break
                    hasil.append(ubah)
                    koreksi += int(ubah != karakter)
                if koreksi < 0:
                    break
            if koreksi >= 0 and PLATE_PATTERN.fullmatch("".join(hasil)):
                usulan.append((koreksi, awal, akhir, "".join(hasil)))
    if not usulan:
        return asli
    _, _, _, terbaik = min(usulan, key=lambda item: (item[0], item[1], item[2]))
    return terbaik


def read_plate(image: np.ndarray, psm: int = 7) -> OcrOutput:
    """Baca satu citar plat yang sudah dibersihkan dan dirapikan.

    Parameter
    ---------
    image : np.ndarray
        Citra plat (BGR, grayscale, atau biner) dengan teks menghadap ke atas.
    psm : int
        Mode segmentasi halaman Tesseract. Nilai 7 (default) untuk satu baris,
        8 untuk satu kata; keduanya dibandingkan oleh pipeline.

    Pengecualian
    ------------
    OcrUnavailableError
        Jika biner Tesseract belum tersedia. Pesannya memberi langkah perbaikan.
    """
    exe = configure_tesseract()
    if exe is None:
        raise OcrUnavailableError(
            "Tesseract belum terdeteksi. Pasang Tesseract OCR, lalu pastikan "
            "tesseract.exe ada di PATH atau di C:\\Program Files\\Tesseract-OCR. "
            "Di Streamlit Cloud, tambahkan `tesseract-ocr` pada packages.txt."
        )

    if image is None or image.size == 0:
        return OcrOutput("", "", 0.0, False)

    try:
        data = pytesseract.image_to_data(
            image,
            config=tesseract_config(psm),
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract tidak ditemukan pada jalur yang terdaftar. Periksa "
            "instalasi Tesseract atau set pytesseract.pytesseract.tesseract_cmd."
        ) from exc
    except Exception as exc:  # pragma: no cover - gabungan error RESTful/exec
        raise OcrUnavailableError(
            f"OCR gagal dijalankan ({type(exc).__name__}: {exc}). Coba lagi "
            "dengan citra yang lebih tajam atau pasang paket bahasa eng."
        ) from exc

    tokens: list[tuple[str, float]] = []
    for i, raw in enumerate(data.get("text", [])):
        token = normalize_plate_text(raw)
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError, KeyError, IndexError):
            conf = 0.0
        if conf > 0:
            tokens.append((token, conf))

    text = "".join(token for token, _ in tokens)
    confidence = (
        float(np.mean([conf for _, conf in tokens])) if tokens else 0.0
    )
    normalized = correct_plate_positions(text)
    return OcrOutput(
        text=text,
        normalized=normalized,
        confidence=round(confidence, 2),
        valid_pattern=validate_plate(normalized),
    )
