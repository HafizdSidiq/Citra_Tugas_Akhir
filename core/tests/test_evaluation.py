"""Tes pembuatan prediksi untuk evaluasi."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core import evaluation
from core.evaluation import _scaled_box, predictions_from_csv, predictions_from_dataset


def test_skala_kotak_kembali_ke_ukuran_asli():
    assert _scaled_box((0, 0, 10, 20), (3024, 3024), (960, 960)) == (0, 0, 32, 63)


def test_prediksi_dari_csv(tmp_path):
    path = tmp_path / "prediksi.csv"
    # Koma pada box perlu dikutip agar CSV tidak terpecah.
    path.write_text(
        'image_id,text,box,confidence\na,B1234ABC,"0,0,10,20",87\n',
        encoding="utf-8",
    )
    result = predictions_from_csv(path)
    assert result["a"][0]["text"] == "B1234ABC"
    assert result["a"][0]["box"] == "0,0,10,20"
    assert result["a"][0]["confidence"] == 87.0


def test_prediksi_dari_dataset_memakai_pipeline(monkeypatch):
    monkeypatch.setattr(evaluation, "get_image_bytes", lambda _: b"gambar")
    plate = SimpleNamespace(
        bbox=(10, 20, 100, 30),
        text="B1234ABC",
        confidence=88.0,
        threshold_used="otsu",
    )
    fake_result = SimpleNamespace(
        stages={
            "original": np.zeros((3024, 3024, 3), dtype=np.uint8),
            "resize": np.zeros((960, 960, 3), dtype=np.uint8),
        },
        plates=[plate],
        timings_ms={"total": 400.0},
    )
    monkeypatch.setattr(evaluation, "process_image", lambda *_: fake_result)

    result = predictions_from_dataset(["test001"], limit=1)
    item = result["test001"][0]
    assert item["text"] == "B1234ABC"
    assert item["box"] == "32,63,346,158"
    assert item["timings_ms"] == {"total": 400.0}
