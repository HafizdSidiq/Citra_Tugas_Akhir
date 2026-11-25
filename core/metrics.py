"""Metrik evaluasi untuk lokalisasi dan pengenalan plat.

Semua angka dihitung hanya dari baris acuan yang benar-benar ada. Tidak ada
prediksi atau label yang diisi otomatis dari nama file/label YOLO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("image_id", "text", "box")


@dataclass(frozen=True)
class EvalResult:
    """Ringkasan hasil evaluasi."""

    n: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    localization: dict[str, float]
    plate_metrics: dict[str, float]
    char_metrics: dict[str, float]
    confusion: pd.DataFrame
    per_condition: pd.DataFrame
    timings: pd.DataFrame
    scenarios: pd.DataFrame
    warnings: tuple[str, ...]


def _parse_box(value: Any) -> tuple[int, int, int, int] | None:
    """Ubah 'x1,y1,x2,y2' menjadi tuple int, atau None bila kosong/invalid."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    parts = str(value).replace(";", ",").split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(part.strip()))) for part in parts)
    except (TypeError, ValueError):
        return None
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _box_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 4:
        return _parse_box(",".join(str(v) for v in value))
    return _parse_box(value)


def iou(box_a: Sequence[int] | str | None, box_b: Sequence[int] | str | None) -> float:
    """Intersection over Union dua kotak (x1, y1, x2, y2)."""
    a, b = _box_tuple(box_a), _box_tuple(box_b)
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def normalize_text(value: Any) -> str:
    """Huruf besar, buang spasi/tanda baca, agar perbandingan adil."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def char_accuracy(reference: str, prediction: str) -> tuple[int, int]:
    """Akurasi per karakter berdasarkan posisi； return (benar, total)."""
    ref, pred = normalize_text(reference), normalize_text(prediction)
    n = max(len(ref), len(pred))
    return sum(1 for i in range(n) if i < len(ref) and i < len(pred) and ref[i] == pred[i]), n


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def _prediction_list(value: Any) -> list[Mapping[str, Any]]:
    """Terima satu prediksi atau banyak prediksi untuk satu image_id."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _split_rows(df: pd.DataFrame, seed: int, test_fraction: float) -> tuple[list[int], list[int]]:
    """Split deterministik tanpa kebocakan: satu image_id hanya di satu split."""
    ids = sorted(str(v) for v in df["image_id"].dropna().unique())
    if len(ids) < 2:
        return [], list(range(len(df)))
    indices = np.arange(len(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test_ids = {ids[i] for i in indices[:n_test]}
    is_test = df["image_id"].astype(str).isin(test_ids).to_numpy()
    return np.flatnonzero(~is_test).tolist(), np.flatnonzero(is_test).tolist()


def load_reference_csv(path: str | Any) -> pd.DataFrame:
    """Baca CSV acuan isian manual dan pastikan kolom minimum ada."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {', '.join(missing)}")
    return df


def evaluate(
    reference: pd.DataFrame | str | Any,
    predictions: Mapping[str, Any],
    *,
    iou_threshold: float = 0.5,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> EvalResult:
    """Hitung metrik dari CSV acuan dan prediksi per image_id.

    Nilai `predictions[image_id]` boleh satu dict atau senarai dict (satu per
    plat yang terdeteksi). Setiap kotak acuan dipasangkan dengan satu kotak
    prediksi terbaik yang belum terpakai, sehingga satu deteksi tidak bisa
    menghitung dua kotak acuan sekaligus. Kunci tambahan yang dikenal:
    `text`, `box`, `confidence`, `condition`, dan `timings_ms`.
    """
    if isinstance(reference, pd.DataFrame):
        df = reference.copy()
    else:
        df = load_reference_csv(reference)
    df = df.reset_index(drop=True)
    train_idx, test_idx = _split_rows(df, seed, test_fraction)
    warnings: list[str] = []
    if not test_idx:
        warnings.append("Hanya satu image_id; semua baris dipakai sebagai data uji.")

    train_ids = tuple(str(df.iloc[i]["image_id"]) for i in train_idx)
    test_ids = tuple(str(df.iloc[i]["image_id"]) for i in test_idx)
    records: list[dict[str, Any]] = []
    used_per_image: dict[str, set[int]] = {}
    missing_preds = 0

    for idx in test_idx:
        row = df.iloc[idx]
        image_id = str(row["image_id"])
        candidates = _prediction_list(predictions.get(image_id))
        ref_box = _parse_box(row.get("box"))
        used = used_per_image.setdefault(image_id, set())
        best_index, best_score, pred = -1, 0.0, {}
        for pos, candidate in enumerate(candidates):
            if pos in used:
                continue
            score = iou(ref_box, candidate.get("box"))
            if score > best_score:
                best_index, best_score, pred = pos, score, candidate
        if best_index >= 0:
            used.add(best_index)
        else:
            missing_preds += 1
        score = best_score
        detected = score >= iou_threshold
        ref_text = normalize_text(row.get("text"))
        pred_text = normalize_text(pred.get("text"))
        correct = bool(ref_text) and detected and ref_text == pred_text
        hits, total_chars = char_accuracy(ref_text, pred_text)
        condition = (
            str(row.get("condition") or "").strip()
            or str(pred.get("condition") or "").strip()
            or "unknown"
        )
        records.append(
            {
                "image_id": image_id,
                "condition": condition,
                "iou": score,
                "detected": detected,
                "plate_correct": correct,
                "char_hits": hits,
                "char_total": total_chars,
                "reference": ref_text,
                "prediction": pred_text,
                "timings_ms": pred.get("timings_ms", {}) or {},
                "confidence": float(pred.get("confidence", 0.0) or 0.0),
            }
        )

    test = pd.DataFrame(records)
    n = len(test)
    detected = int(test["detected"].sum()) if n else 0
    tp = int((test["detected"] & test["plate_correct"]).sum()) if n else 0
    localization = {
        "iou_rata_rata": _mean(test["iou"]) if n else 0.0,
        "presisi": float(tp / detected) if detected else 0.0,
        "recall": float(tp / n) if n else 0.0,
        "deteksi": float(detected / n) if n else 0.0,
        "ambang_iou": float(iou_threshold),
    }
    f1_den = localization["presisi"] + localization["recall"]
    localization["f1"] = float(2 * localization["presisi"] * localization["recall"] / f1_den) if f1_den else 0.0

    plate_acc = float(tp / n) if n else 0.0
    char_hits = int(test["char_hits"].sum()) if n else 0
    char_total = int(test["char_total"].sum()) if n else 0
    plate_metrics = {
        "akurasi_plat": plate_acc,
        "akurasi_karakter": float(char_hits / char_total) if char_total else 0.0,
        "total_karakter_acuan": char_total,
        "karakter_benar": char_hits,
    }

    labels = sorted(set(test["reference"]) | set(test["prediction"])) if n else []
    index = labels if labels else ["—"]
    confusion = pd.DataFrame(0, index=index, columns=index, dtype=int)
    if n:
        for ref, pred in zip(test["reference"], test["prediction"]):
            confusion.loc[ref, pred] += 1

    per_condition = (
        test.groupby("condition", as_index=False)
        .agg(
            n=("image_id", "size"),
            iou=("iou", "mean"),
            deteksi=("detected", "mean"),
            akurasi_plat=("plate_correct", "mean"),
            akurasi_karakter=("char_hits", "sum"),
            total_karakter=("char_total", "sum"),
        )
        if n
        else pd.DataFrame()
    )
    if not per_condition.empty:
        per_condition["akurasi_karakter"] = per_condition.apply(
            lambda row: row["akurasi_karakter"] / row["total_karakter"]
            if row["total_karakter"]
            else 0.0,
            axis=1,
        )

    timing_rows: dict[str, list[float]] = {}
    timing_images: set[str] = set()
    for record in records:
        if record["image_id"] in timing_images:
            continue
        timing_images.add(record["image_id"])
        for stage, value in (record["timings_ms"] or {}).items():
            try:
                timing_rows.setdefault(str(stage), []).append(float(value))
            except (TypeError, ValueError):
                continue
    timings = (
        pd.DataFrame(
            [
                {"Tahap": stage, "Rata-rata (ms)": _mean(values), "N": len(values)}
                for stage, values in sorted(timing_rows.items())
            ]
        )
        if timing_rows
        else pd.DataFrame(columns=["Tahap", "Rata-rata (ms)", "N"])
    )

    scenario_specs = [
        ("siang", "Siang"),
        ("malam", "Malam"),
        ("bayangan", "Bayangan"),
        ("tegak", "Tegak"),
        ("miring", "Miring"),
        ("jauh", "Jauh"),
        ("mobil", "Mobil"),
        ("motor", "Motor"),
        ("clahe", "CLAHE aktif"),
        ("tanpa_clahe", "Tanpa CLAHE"),
        ("otsu", "Otsu"),
        ("adaptif", "Adaptif"),
    ]
    scenario_rows = []
    for key, label in scenario_specs:
        subset = test[test["condition"].str.lower() == key] if n else test
        scenario_rows.append(
            {
                "Skenario": label,
                "N": len(subset),
                "IoU": _mean(subset["iou"]) if len(subset) else 0.0,
                "Akurasi plat": _mean(subset["plate_correct"]) if len(subset) else 0.0,
                "Status": "Terukur" if len(subset) else "Belum ada data",
            }
        )
    scenarios = pd.DataFrame(scenario_rows)

    test_reference = df.loc[test_idx]
    unfilled = int((test_reference["text"].astype(str).str.strip() == "").sum())
    missing_boxes = int((test_reference["box"].astype(str).str.strip() == "").sum())
    if unfilled:
        warnings.append(f"{unfilled} baris uji belum memiliki teks acuan; baris itu tidak dihitung benar.")
    if missing_boxes:
        warnings.append(f"{missing_boxes} baris uji belum memiliki kotak acuan; IoU baris itu = 0.")
    if missing_preds:
        warnings.append(f"{missing_preds} kotak acuan tidak mendapat prediksi yang cocok; dihitung sebagai miss.")

    return EvalResult(
        n=n,
        train_ids=train_ids,
        test_ids=test_ids,
        localization=localization,
        plate_metrics=plate_metrics,
        char_metrics={
            "akurasi": plate_metrics["akurasi_karakter"],
            "benar": char_hits,
            "total": char_total,
        },
        confusion=confusion,
        per_condition=per_condition,
        timings=timings,
        scenarios=scenarios,
        warnings=tuple(warnings),
    )


def reference_template(path: str, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Buat template CSV acuan kosong; tidak mengisi teks otomatis."""
    names = [f"test{i:03d}" for i in range(1, n + 1)]
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    df = pd.DataFrame(
        {
            "image_id": names,
            "text": [""] * n,
            "box": [""] * n,
            "condition": [""] * n,
        }
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df
