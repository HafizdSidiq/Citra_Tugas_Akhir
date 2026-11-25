"""Akses ringan ke dataset Kaggle untuk galeri contoh.

Hanya file yang dibutuhkan yang diambil. Seluruh arsip ~2 GB tidak pernah
diunduh. Kredensial, bila ada, dibaca dari st.secrets dan tidak ditulis ke disk.
"""

from __future__ import annotations

import csv
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import streamlit as st

DATASET_HANDLE = "juanthomaswijaya/indonesian-license-plate-dataset"
IMAGE_TEMPLATE = "Indonesian License Plate Dataset/images/test/{name}.jpg"
LABEL_TEMPLATE = "Indonesian License Plate Dataset/labelswithLP/test/{name}.txt"
TOTAL_TEST_IMAGES = 100


class DatasetError(RuntimeError):
    """Gagal mengakses dataset, dengan saran perbaikan."""


def configure_kaggle_credentials() -> None:
    """Ambil kredensial dari st.secrets bila tersedia."""
    secrets: dict[str, Any] = {}
    try:
        secrets = dict(st.secrets.get("kaggle", {}))  # type: ignore[arg-type]
    except Exception:
        secrets = {}

    username = secrets.get("username") or secrets.get("KAGGLE_USERNAME")
    key = secrets.get("key") or secrets.get("KAGGLE_KEY")
    if username:
        os.environ.setdefault("KAGGLE_USERNAME", str(username))
    if key:
        os.environ.setdefault("KAGGLE_KEY", str(key))


@st.cache_data(ttl=3600, max_entries=50)
def dataset_image_ids(seed: int = 42, limit: int = 100) -> tuple[str, ...]:
    """Pilih nama file uji dengan seed tetap, tanpa mengunduh citra."""
    limit = max(0, min(int(limit), TOTAL_TEST_IMAGES))
    names = [f"test{i:03d}" for i in range(1, TOTAL_TEST_IMAGES + 1)]
    random.Random(int(seed)).shuffle(names)
    return tuple(names[:limit])


@st.cache_data(ttl=3600, max_entries=50)
def gallery_items(seed: int = 42, limit: int = 24) -> tuple[dict[str, str], ...]:
    """Pilih nama file galeri dengan seed tetap (maksimal 24)."""
    return tuple(
        {
            "label": name,
            "image_path": image_path_for(name),
            "label_path": LABEL_TEMPLATE.format(name=name),
        }
        for name in dataset_image_ids(seed, limit)
    )


def image_path_for(name: str) -> str:
    """Path satu citra di dalam dataset."""
    return IMAGE_TEMPLATE.format(name=str(name))


def label_path_for(name: str) -> str:
    """Path satu label dengan teks plat di dalam dataset."""
    return LABEL_TEMPLATE.format(name=str(name))


def reference_template_rows(seed: int = 42, limit: int = 100) -> tuple[dict[str, str], ...]:
    """Buat baris acuan: kotak dari label YOLO, teks plat sengaja dikosongkan.

    Kolom keenam labelswithLP tidak dipakai sebagai teks acuan sesuai ketentuan
    tugas; teks harus diketik manual oleh pengguna.
    """
    names = dataset_image_ids(seed, limit)
    if not names:
        return ()
    with Image.open(BytesIO(get_image_bytes(image_path_for(names[0])))) as image:
        width, height = image.size

    rows: list[dict[str, str]] = []
    for name in names:
        content = get_label_text(label_path_for(name))
        boxes: list[str] = []
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, bw, bh = (float(part) for part in parts[1:5])
            except ValueError:
                continue
            x1 = max(0, min(width, round((xc - bw / 2) * width)))
            y1 = max(0, min(height, round((yc - bh / 2) * height)))
            x2 = max(0, min(width, round((xc + bw / 2) * width)))
            y2 = max(0, min(height, round((yc + bh / 2) * height)))
            if x2 > x1 and y2 > y1:
                boxes.append(f"{x1},{y1},{x2},{y2}")
        for box in boxes or [""]:
            rows.append({"image_id": name, "text": "", "box": box, "condition": ""})
    return tuple(rows)


def write_reference_template(
    path: str | Path, seed: int = 42, limit: int = 100
) -> tuple[dict[str, str], ...]:
    """Tulis template CSV acuan; aman ditimpa karena teks acuan tetap kosong."""
    rows = reference_template_rows(seed, limit)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "text", "box", "condition"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


@st.cache_data(ttl=3600, max_entries=50)
def get_image_bytes(image_path: str) -> bytes:
    """Unduh satu citra sesuai path dan kembalikan bytes di memori."""
    configure_kaggle_credentials()
    try:
        import kagglehub
    except ImportError as exc:
        raise DatasetError(
            "Paket kagglehub belum terpasang. Tambahkan `kagglehub` ke requirements.txt."
        ) from exc
    try:
        local = Path(kagglehub.dataset_download(DATASET_HANDLE, path=image_path))
        return local.read_bytes()
    except Exception as exc:
        raise DatasetError(
            "Gagal mengambil citra dari Kaggle. Periksa koneksi internet dan, bila "
            "diperlukan, isi username/key Kaggle pada st.secrets. Unggah foto lokal "
            "tetap bisa digunakan."
        ) from exc


@st.cache_data(ttl=3600, max_entries=50)
def get_label_text(label_path: str) -> str:
    """Ambil label YOLO satu file untuk keperluan evaluasi."""
    configure_kaggle_credentials()
    try:
        import kagglehub
    except ImportError as exc:
        raise DatasetError("Paket kagglehub belum terpasang.") from exc
    try:
        local = Path(kagglehub.dataset_download(DATASET_HANDLE, path=label_path))
        return local.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise DatasetError(
            "Gagal mengambil label dari Kaggle. Periksa koneksi internet atau kredensial."
        ) from exc


def dataset_status() -> dict[str, Any]:
    """Cek koneksi dengan mengambil satu citra kecil dari galeri."""
    items = gallery_items(seed=42, limit=1)
    try:
        payload = get_image_bytes(items[0]["image_path"])
        return {
            "available": True,
            "message": "Terhubung ke Kaggle. Galeri contoh siap dipakai.",
            "sample_bytes": len(payload),
        }
    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
            "sample_bytes": 0,
        }
