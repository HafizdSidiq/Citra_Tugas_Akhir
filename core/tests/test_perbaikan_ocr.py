"""Tes untuk koreksi posisi OCR, filter luas, dan urutan skor hasil."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.ocr import correct_plate_positions
from core.pipeline import (
    PipelineParams,
    crop_plate_for_ocr,
    detect_edges,
    enhance_image,
    find_plate_candidates,
    process_image,
    score_candidate,
)


@pytest.mark.parametrize(
    ("masukan", "diharapkan"),
    [
        ("B1234ABC", "B1234ABC"),
        ("B1234A8C", "B1234ABC"),
        ("B1O23ABC", "B1023ABC"),
        ("8I234ABC", "BI234ABC"),
        ("B1234AB1", "B1234ABI"),
        ("B1234ABX", "B1234ABX"),
    ],
)
def test_koreksi_posisi_memilih_jumlah_ubah_paling_sedikit(
    masukan: str, diharapkan: str
) -> None:
    assert correct_plate_positions(masukan) == diharapkan


def test_koreksi_tidak_memaksa_pasangan_yang_tidak_dikenal() -> None:
    assert correct_plate_positions("B1234ABX") == "B1234ABX"
    assert correct_plate_positions("B1234ABC1") == "B1234ABC1"


def _gambar_satu_persegi_panjang() -> np.ndarray:
    gambar = np.full((400, 600, 3), 200, dtype=np.uint8)
    cv2.rectangle(gambar, (100, 150), (400, 250), (255, 255, 255), -1)
    cv2.rectangle(gambar, (100, 150), (400, 250), (20, 20, 20), 3)
    return gambar


def test_luas_maksimum_membuang_kontur_besar() -> None:
    _, enhanced = enhance_image(_gambar_satu_persegi_panjang())
    edges = detect_edges(enhanced)
    assert find_plate_candidates(edges, PipelineParams())
    assert find_plate_candidates(edges, PipelineParams(max_area_ratio=0.05)) == []


def test_pangkas_bawah_membuang_baris_masa_berlaku() -> None:
    plat = np.full((200, 600, 3), 255, dtype=np.uint8)
    plat[150:, :] = 0
    potongan = crop_plate_for_ocr(plat, bottom_ratio=0.25, side_ratio=0.03)
    assert potongan.shape == (144, 564, 3)
    assert float(potongan.mean()) == 255.0


def test_skor_mengutamakan_pola_dan_teks_valid() -> None:
    valid = score_candidate("B1234ABC", 90.0, 7, 0.5, 2.8)
    invalid = score_candidate("XX", 90.0, 2, 0.1, 1.4)
    kosong = score_candidate("", 100.0, 7, 0.5, 2.8)
    assert valid > invalid > kosong
    assert kosong == 0.0


def test_hasil_pipeline_diurutkan_dari_skor_tertinggi() -> None:
    hasil = process_image(_gambar_satu_persegi_panjang(), PipelineParams(min_score=0))
    skor = [plate.score for plate in hasil.plates]
    assert skor == sorted(skor, reverse=True)
    assert hasil.candidate_boxes
