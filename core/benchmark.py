"""Patok ukur lokalisasi dan bacaan plat pada data uji asli.

Jalankan dari akar repo, sebelum dan sesudah mengubah pipeline:

    python -m core.benchmark sebelum 60
    python -m core.benchmark sesudah 60

Angka disimpan di docs/benchmark/<label>.json. Kotak acuan berasal dari
docs/acuan_ocr.csv (label YOLO dataset). Kolom `text` CSV sengaja tidak diubah;
untuk metrik teks, acuan diambil dari kolom ke-6 label YOLO yang sudah ada di
cache dataset. Sumber ini dicatat di berkas hasil agar tidak tertukar dengan
teks acuan buatan pengguna.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import (
    DatasetError,
    get_image_bytes,
    get_label_text,
    image_path_for,
    label_path_for,
)
from .metrics import char_accuracy, iou, normalize_text
from .pipeline import PipelineParams, process_image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "acuan_ocr.csv"
OUTPUT = ROOT / "docs" / "benchmark"
IOU_THRESHOLD = 0.5


def _reference_groups(limit: int) -> list[tuple[str, list[dict[str, str]]]]:
    with REFERENCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for row in rows:
        name = str(row.get("image_id") or "").strip()
        if not name or not str(row.get("box") or "").strip():
            continue
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(row)
    return [(name, groups[name]) for name in order[:limit]]


def _to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def _label_texts(name: str) -> list[str]:
    """Teks plat dari kolom keenam label YOLO (bukan dari CSV acuan)."""
    try:
        isi = get_label_text(label_path_for(name))
    except Exception:
        return []
    hasil = []
    for baris in isi.splitlines():
        bagian = baris.split()
        if len(bagian) >= 6:
            hasil.append(normalize_text(bagian[5]))
    return hasil


def run(limit: int = 60, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    groups = _reference_groups(limit)
    if len(groups) < 50:
        raise RuntimeError(f"Baru {len(groups)} citra acuan; minimal 50 diperlukan.")

    param = PipelineParams()
    for kunci, nilai in (overrides or {}).items():
        if not hasattr(param, kunci):
            raise ValueError(f"Parameter tidak dikenal: {kunci}")
        setattr(param, kunci, nilai)
    n_refs = 0
    ref_hits = 0
    top1_hits = 0
    boxes_total = 0
    n_with_text = 0
    plate_correct = 0
    char_hits_total = 0
    char_total = 0
    dipakai: list[str] = []
    gagal: list[str] = []

    for name, rows in groups:
        refs = [
            tuple(int(round(float(v))) for v in str(row["box"]).replace(";", ",").split(","))
            for row in rows
        ]
        refs = [ref for ref in refs if len(ref) == 4]
        if not refs:
            continue
        n_refs += len(refs)

        try:
            payload = get_image_bytes(image_path_for(name))
            result = process_image(payload, param)
        except (DatasetError, Exception) as exc:  # pragma: no cover - bergantung jaringan
            gagal.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        dipakai.append(name)

        asli = result.stages["original"]
        resize = result.stages["resize"]
        scale = float(resize.shape[1]) / float(asli.shape[1])
        # Recall memakai semua kandidat geometri, bukan hanya yang lolos OCR.
        mentah = getattr(result, "candidate_boxes", None)
        if mentah is None:  # kompatibel dengan hasil sebelum pembaruan
            mentah = [plate.bbox for plate in result.plates]
        kandidat = [_to_xyxy(tuple(float(v) / scale for v in box)) for box in mentah]
        final = [
            _to_xyxy(tuple(float(v) / scale for v in plate.bbox))
            for plate in result.plates
        ]
        boxes_total += len(final)

        ref_hits += sum(
            1
            for ref in refs
            if any(iou(ref, cand) >= IOU_THRESHOLD for cand in kandidat)
        )
        if final:
            primary_iou = max(iou(ref, final[0]) for ref in refs)
            top1_hits += int(primary_iou >= IOU_THRESHOLD)

        label_texts = _label_texts(name)
        for row, ref, teks_label in zip(rows, refs, label_texts):
            teks = normalize_text(row.get("text")) or teks_label
            if not teks:
                continue
            n_with_text += 1
            if not final:
                continue
            pos = max(range(len(final)), key=lambda i: iou(ref, final[i]))
            if iou(ref, final[pos]) < IOU_THRESHOLD:
                continue
            pred = normalize_text(result.plates[pos].text)
            plate_correct += int(pred == teks)
            hits, total = char_accuracy(teks, pred)
            char_hits_total += hits
            char_total += total

    n = len(dipakai)
    if n == 0:
        raise RuntimeError("Tidak ada citra yang berhasil diproses.")
    ringkasan = {
        "waktu": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sumber": "Kaggle juanthomaswijaya/indonesian-license-plate-dataset, split test",
        "acuan": str(REFERENCE.relative_to(ROOT)),
        "sumber_teks": "Kolom ke-6 labelswithLP, dipakai hanya untuk metrik teks; CSV acuan tidak diubah",
        "n_citra_diminta": len(groups),
        "n_citra_dipakai": n,
        "n_kotak_acuan": n_refs,
        "ambang_iou": IOU_THRESHOLD,
        "recall_lokalisasi": round(ref_hits / n_refs, 4) if n_refs else 0.0,
        "hit_top1": round(top1_hits / n, 4),
        "rata_kotak_per_citra": round(boxes_total / n, 3),
        "n_teks_acuan": n_with_text,
        "akurasi_plat": round(plate_correct / n_with_text, 4) if n_with_text else None,
        "akurasi_karakter": round(char_hits_total / char_total, 4) if char_total else None,
        "params": param.__dict__,
        "gagal": gagal[:10],
    }
    return ringkasan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Nama berkas hasil, mis. sebelum atau sesudah")
    parser.add_argument("jumlah", nargs="?", type=int, default=60, help="Jumlah citra (minimal 50)")
    parser.add_argument(
        "--params",
        default="{}",
        help='Override JSON, mis. \'{"psm_modes":[7],"min_score":0}\'',
    )
    args = parser.parse_args()
    overrides = json.loads(args.params)
    if not isinstance(overrides, dict):
        raise ValueError("--params harus berupa objek JSON")
    hasil = run(args.jumlah, overrides)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tujuan = OUTPUT / f"{args.label}.json"
    tujuan.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(f"Tersimpan: {tujuan.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
