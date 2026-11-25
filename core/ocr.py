"""Pembacaan teks plat dengan Tesseract OCR.

Semua konfigurasi OCR bersifat eksplisit agar mudah dijelaskan: OCR memakai
whitelist A-Z0-9 dan page segmentation mode 7 (satu baris teks).
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
TESSERACT_CONFIG = f"--oem 3 --psm 7 -c tessedit_char_whitelist={WHITELIST}"


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


def read_plate(image: np.ndarray) -> OcrOutput:
    """Baca satu citar plat yang sudah dibersihkan dan dirapikan.

    Parameter
    ---------
    image : np.ndarray
        Citra plat (BGR, grayscale, atau biner) dengan teks menghadap ke atas.

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
            config=TESSERACT_CONFIG,
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
    return OcrOutput(
        text=text,
        normalized=text,
        confidence=round(confidence, 2),
        valid_pattern=validate_plate(text),
    )
