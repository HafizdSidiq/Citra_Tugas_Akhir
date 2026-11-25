"""Pembuatan prediksi untuk evaluasi dari citra dataset.

Prediksi dihasilkan oleh pipeline yang sama dengan halaman Deteksi. Teks acuan
tidak pernah diambil dari dataset; hanya kotak label YOLO yang dipakai template.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .dataset import get_image_bytes, image_path_for
from .pipeline import PipelineParams, process_image


def _scaled_box(
    box: tuple[int, int, int, int], original: tuple[int, int], resized: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Kembalikan kotak dari citra resize ke koordinat citra asli."""
    scale_x = original[1] / float(max(1, resized[1]))
    scale_y = original[0] / float(max(1, resized[0]))
    x1, y1, x2, y2 = box
    return (
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    )


def predictions_from_csv(path: Any) -> dict[str, list[dict[str, Any]]]:
    """Baca prediksi dari CSV dengan kolom image_id, text, box."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [col for col in ("image_id", "text", "box") if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom prediksi wajib hilang: {', '.join(missing)}")

    result: dict[str, list[dict[str, Any]]] = {}
    for row in df.to_dict("records"):
        image_id = str(row["image_id"]).strip()
        if not image_id:
            continue
        try:
            confidence = float(row.get("confidence", "") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        result.setdefault(image_id, []).append(
            {
                "text": str(row.get("text", "")),
                "box": str(row.get("box", "")),
                "confidence": confidence,
                "condition": str(row.get("condition", "")),
            }
        )
    return result


def predictions_from_dataset(
    image_ids: Sequence[str],
    params: PipelineParams | None = None,
    conditions: Mapping[str, str] | None = None,
    limit: int = 24,
) -> dict[str, list[dict[str, Any]]]:
    """Jalankan pipeline pada beberapa citra uji dan kumpulkan prediksinya."""
    params = params or PipelineParams()
    result: dict[str, list[dict[str, Any]]] = {}
    conditions = conditions or {}

    for name in list(image_ids)[: max(0, int(limit))]:
        payload = get_image_bytes(image_path_for(name))
        pipeline_result = process_image(payload, params)
        original = pipeline_result.stages["original"].shape[1::-1]  # (lebar, tinggi)
        resized_stage = pipeline_result.stages.get("resize", pipeline_result.stages["original"])
        resized = resized_stage.shape[1::-1]
        items: list[dict[str, Any]] = []
        for plate in pipeline_result.plates:
            # cv2.boundingRect mengembalikan (x, y, w, h), bukan (x1, y1, x2, y2).
            x, y, w, h = plate.bbox
            x1, y1, x2, y2 = _scaled_box((x, y, x + w, y + h), original, resized)
            items.append(
                {
                    "text": plate.text,
                    "box": f"{x1},{y1},{x2},{y2}",
                    "confidence": round(float(plate.confidence), 2),
                    "condition": str(conditions.get(name, "")),
                    "threshold_used": plate.threshold_used,
                    "timings_ms": dict(pipeline_result.timings_ms),
                }
            )
        result[str(name)] = items
    return result
