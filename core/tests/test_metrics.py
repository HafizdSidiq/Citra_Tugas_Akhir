"""Tes metrik evaluasi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.metrics import (
    char_accuracy,
    evaluate,
    iou,
    normalize_text,
    reference_template,
)


def test_iou_sempurna_dan_tidak_beririsan():
    assert iou("0,0,10,10", "0,0,10,10") == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou("", "0,0,10,10") == 0.0


def test_iou_setengah():
    # Luas masing-masing 5000; irisan 25x50 = 1250; union = 8750.
    assert iou("0,0,50,100", "25,50,75,150") == pytest.approx(1250 / 8750)


def test_normalisasi_teks():
    assert normalize_text(" b 1234 abc ") == "B1234ABC"
    assert normalize_text(None) == ""


def test_akurasi_karakter_posisional():
    assert char_accuracy("B1234ABC", "B1234ABC") == (8, 8)
    assert char_accuracy("B1234ABC", "B1234ABD") == (7, 8)
    assert char_accuracy("B1234ABC", "B123") == (4, 8)


def test_evaluasi_penuh():
    df = pd.DataFrame(
        {
            "image_id": [f"test{i:03d}" for i in range(6)],
            "text": ["B1234ABC"] * 6,
            "box": ["10,10,110,50"] * 6,
            "condition": ["siang", "malam", "bayangan", "miring", "mobil", "motor"],
        }
    )
    pred = {
        row["image_id"]: {
            "text": "B1234ABC" if i % 2 == 0 else "B1234ABD",
            "box": "10,10,110,50" if i < 5 else "200,200,300,240",
            "confidence": 87,
            "timings_ms": {"ocr": 100 + i, "total": 200 + i},
        }
        for i, (_, row) in enumerate(df.iterrows())
    }
    result = evaluate(df, pred, iou_threshold=0.5, seed=42, test_fraction=0.34)
    assert result.n == 2
    assert 0.0 <= result.localization["iou_rata_rata"] <= 1.0
    assert 0.0 <= result.plate_metrics["akurasi_plat"] <= 1.0
    assert result.char_metrics["total"] >= 0
    assert list(result.confusion.index) == list(result.confusion.columns)
    assert set(result.timings["Tahap"]) == {"ocr", "total"}
    assert len(result.scenarios) == 12


def test_beberapa_kotak_per_citra_tidak_dipakai_ulang():
    df = pd.DataFrame(
        {
            "image_id": ["a", "a"],
            "text": ["A1", "B2"],
            "box": ["0,0,100,20", "0,30,100,50"],
            "condition": ["siang", "siang"],
        }
    )
    pred = {
        "a": [
            {"text": "B2", "box": "0,30,100,50", "confidence": 80},
            {"text": "A1", "box": "0,0,100,20", "confidence": 70},
        ]
    }
    result = evaluate(df, pred, seed=3, test_fraction=1.0)
    assert result.n == 2
    assert result.plate_metrics["akurasi_plat"] == 1.0
    assert result.localization["iou_rata_rata"] == 1.0


def test_evaluasi_tanpa_prediksi_tetap_aman():
    df = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "b"],
            "text": ["A1", "A1", "B2", "B2"],
            "box": ["0,0,10,10", "0,0,10,10", "0,0,10,10", "0,0,10,10"],
            "condition": ["siang", "siang", "malam", "malam"],
        }
    )
    result = evaluate(df, {}, seed=42, test_fraction=0.5)
    assert result.n >= 1
    assert result.plate_metrics["akurasi_plat"] == 0.0
    assert any("tidak mendapat prediksi" in w for w in result.warnings)


def test_split_tidak_bocor_per_image_id():
    df = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "b", "c", "c"],
            "text": ["A1", "A1", "B2", "B2", "C3", "C3"],
            "box": ["0,0,10,10"] * 6,
            "condition": ["siang"] * 6,
        }
    )
    result = evaluate(df, {}, seed=7, test_fraction=0.34)
    # Satu image_id masuk uji, tetapi dua barisnya ikut semua => tidak bocor ke latih.
    assert result.n == 2
    assert len(set(result.test_ids)) == 1
    assert not set(result.test_ids) & set(result.train_ids)


def test_template_kosong(tmp_path):
    path = tmp_path / "acuan.csv"
    df = reference_template(str(path), n=5)
    assert len(df) == 5
    assert set(df["text"]) == {""}
    assert pd.read_csv(path, dtype=str)["image_id"].tolist() == df["image_id"].tolist()
