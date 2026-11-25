"""Tes unit alur pengolahan citra dengan citra sintetis di memori."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.ocr import configure_tesseract, read_plate, validate_plate
from core.pipeline import (
    MAX_WIDTH,
    RECT_MIN_HEIGHT,
    PipelineParams,
    binarize_plate,
    detect_edges,
    enhance_image,
    find_plate_candidates,
    load_image,
    perspective_rectify,
    process_image,
    resize_image,
    segment_characters,
)


def make_plate_image(
    text: str = "B1234ABC", size: tuple[int, int] = (1200, 800), angle: float = 0.0
) -> np.ndarray:
    """Buat foto sintetis berisi satu plat putih dengan tulisan hitam."""
    width, height = size
    gradient = np.linspace(185, 225, width, dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        image[:, :, channel] = gradient

    plate_w, plate_h = 560, 160
    x0, y0 = (width - plate_w) // 2, (height - plate_h) // 2
    cv2.rectangle(image, (x0, y0), (x0 + plate_w, y0 + plate_h), (255, 255, 255), -1)
    cv2.rectangle(image, (x0, y0), (x0 + plate_w, y0 + plate_h), (20, 20, 20), 4)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    cv2.putText(
        image,
        text,
        (x0 + (plate_w - tw) // 2, y0 + (plate_h + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (15, 15, 15),
        3,
        cv2.LINE_AA,
    )

    if angle:
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=(205, 205, 205))
    return image


def test_resize_membatasi_lebar_dan_menjaga_rasio() -> None:
    image = np.zeros((400, 1600, 3), dtype=np.uint8)
    out = resize_image(image, MAX_WIDTH)
    assert out.shape[1] == MAX_WIDTH
    assert out.shape[0] == 240
    unchanged = resize_image(image, 2000)
    assert unchanged.shape == image.shape
    assert unchanged is not image


def test_enhance_menghasilkan_abu_abu_dan_hasil_filter() -> None:
    image = make_plate_image()
    gray, enhanced = enhance_image(image, use_clahe=True)
    assert gray.ndim == enhanced.ndim == 2
    assert not np.array_equal(gray, enhanced)
    _, without_clahe = enhance_image(image, use_clahe=False)
    assert not np.array_equal(enhanced, without_clahe)


def test_edges_mendeteksi_tepi_plat() -> None:
    image = make_plate_image()
    _, enhanced = enhance_image(image)
    edges = detect_edges(enhanced, 50, 150)
    assert edges.ndim == 2
    assert int(np.count_nonzero(edges)) > 1000


def test_kandidat_plat_ditemukan_pada_citra_miring() -> None:
    image = make_plate_image(angle=8)
    _, enhanced = enhance_image(image)
    edges = detect_edges(enhanced)
    quads = find_plate_candidates(edges)
    assert len(quads) >= 1
    x, y, w, h = cv2.boundingRect(np.round(quads[0]).astype(np.int32))
    assert w > h
    assert 1.6 <= w / h <= 3.8


def test_binarisasi_menghasilkan_dua_varian() -> None:
    image = make_plate_image()
    _, enhanced = enhance_image(image)
    quad = find_plate_candidates(detect_edges(enhanced))[0]
    rectified = perspective_rectify(image, quad)
    otsu, adaptive = binarize_plate(rectified)
    assert otsu.shape == adaptive.shape
    assert otsu.shape[0] >= RECT_MIN_HEIGHT
    assert 2.0 <= otsu.shape[1] / otsu.shape[0] <= 3.5
    assert otsu.mean() > 127 and adaptive.mean() > 127


def test_segmentasi_karakter_diurutkan_kiri_ke_kanan() -> None:
    image = make_plate_image()
    _, enhanced = enhance_image(image)
    quad = find_plate_candidates(detect_edges(enhanced))[0]
    otsu, _ = binarize_plate(perspective_rectify(image, quad))
    boxes = segment_characters(otsu)
    assert len(boxes) >= 5
    assert [box[0] for box in boxes] == sorted(box[0] for box in boxes)


def test_validasi_pola_plat() -> None:
    assert validate_plate("B1234ABC")
    assert validate_plate("S61310CF")
    assert validate_plate("b 1234 abc")
    assert not validate_plate("B1234AB1")
    assert not validate_plate("1234ABC")


def test_pipeline_mengembalikan_tahap_waktu_dan_kotak() -> None:
    image = make_plate_image()
    result = process_image(image, PipelineParams(canny_low=50, canny_high=150))
    expected = {"original", "resize", "grayscale", "enhanced", "edges", "contours", "annotated"}
    assert expected.issubset(result.stages)
    assert result.annotated_image.shape[:2] == result.stages["resize"].shape[:2]
    assert len(result.plates) >= 1
    assert {"resize", "enhance", "edges", "contours", "ocr", "total"}.issubset(result.timings_ms)
    assert result.timings_ms["total"] > 0
    assert result.plates[0].threshold_used in {"otsu", "adaptive"}


def test_pipeline_tanpa_plat_memberi_pesan_perbaikan() -> None:
    blank = np.full((600, 900, 3), 235, dtype=np.uint8)
    result = process_image(blank, PipelineParams(min_area_ratio=0.02))
    assert result.plates == []
    assert result.text == ""
    assert any("plat terlalu kecil" in w.lower() for w in result.warnings)


def test_load_image_menerima_bytes_dan_pil() -> None:
    from io import BytesIO

    from PIL import Image

    ok, encoded = cv2.imencode(".png", make_plate_image(size=(400, 300)))
    assert ok
    from_bytes = load_image(encoded.tobytes())
    assert from_bytes.shape[:2] == (300, 400)
    pil = Image.open(BytesIO(encoded.tobytes())).convert("RGB")
    assert load_image(pil).shape[:2] == (300, 400)


@pytest.mark.skipif(configure_tesseract() is None, reason="Tesseract tidak terpasang")
def test_ocr_tesseract_berjalan_pada_citra_sintetis() -> None:
    image = make_plate_image()
    _, enhanced = enhance_image(image)
    quad = find_plate_candidates(detect_edges(enhanced))[0]
    otsu, _ = binarize_plate(perspective_rectify(image, quad))
    boxes = segment_characters(otsu)
    from core.pipeline import rebuild_line

    output = read_plate(rebuild_line(otsu, boxes) if boxes else otsu)
    assert 0.0 <= output.confidence <= 100.0
    assert output.normalized == output.normalized.upper()
    assert output.valid_pattern == validate_plate(output.normalized)
