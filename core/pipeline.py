"""Alur pengolahan citra klasik untuk membaca plat nomor Indonesia.

Setiap tahap mengembalikan citra antara sehingga semua langkah bisa ditampilkan
dan dijelaskan pada halaman Proses. Tidak ada model deep learning di sini.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .ocr import OcrUnavailableError, read_plate, validate_plate

MAX_WIDTH = 960
RECT_SIZE = (600, 180)
PREFERRED_RATIO = 2.8


@dataclass
class PipelineParams:
    """Parameter yang bisa diubah pengguna pada halaman Proses."""

    canny_low: int = 50
    canny_high: int = 150
    min_area_ratio: float = 0.001
    aspect_min: float = 1.6
    aspect_max: float = 3.8
    max_candidates: int = 6
    use_clahe: bool = True
    binarization: str = "auto"  # "auto", "otsu", atau "adaptive"


@dataclass
class PlateCandidate:
    """Satu plat yang berhasil dilokalisasi dan dibaca."""

    index: int
    quad: np.ndarray
    bbox: tuple[int, int, int, int]
    rectified: np.ndarray
    otsu: np.ndarray
    adaptive: np.ndarray
    binary: np.ndarray
    threshold_used: str
    char_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0
    valid_pattern: bool = False
    score: float = 0.0


@dataclass
class PipelineResult:
    """Keluaran lengkap satu kali pemrosesan."""

    stages: dict[str, np.ndarray]
    annotated_image: np.ndarray
    plates: list[PlateCandidate]
    text: str
    texts: list[str]
    confidence: float
    timings_ms: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def load_image(source: Any) -> np.ndarray:
    """Ubah unggahan/PIL/array menjadi citra BGR di memori."""
    if isinstance(source, bytes):
        array = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            with Image.open(__import__("io").BytesIO(source)) as pil:
                image = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return image
    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if source.ndim == 3 and source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
        if source.ndim == 3 and source.shape[2] == 3:
            return source.copy()
    raise TypeError("Format citra tidak dikenal. Unggah JPG, PNG, atau WEBP.")


def resize_image(image: np.ndarray, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Tahap 1: perkecil lebar maksimum, rasio aspek tetap."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()
    scale = max_width / float(width)
    size = (max_width, max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def enhance_image(
    image: np.ndarray, use_clahe: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Tahap 2: grayscale -> CLAHE -> bilateral filter.

    Mengembalikan (abu-abu mentah, hasil peningkatan) agar efek CLAHE bisa
    dibandingkan pada halaman Evaluasi.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if use_clahe else gray
    enhanced = cv2.bilateralFilter(base, d=5, sigmaColor=50, sigmaSpace=50)
    return gray, enhanced


def detect_edges(
    enhanced: np.ndarray, low: int = 50, high: int = 150
) -> np.ndarray:
    """Tahap 3: Canny lalu closing morfologi untuk menyambung tepi plat."""
    low, high = int(low), int(high)
    if low > high:
        low, high = high, low
    edges = cv2.Canny(enhanced, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def _order_points(points: np.ndarray) -> np.ndarray:
    """Urutkan 4 titik: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    diffs = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU dua kotak (x, y, w, h), dipakai untuk menghapus duplikat kontur."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def find_plate_candidates(
    edges: np.ndarray, params: PipelineParams | None = None
) -> list[np.ndarray]:
    """Tahap 4: saring kontur persegi panjang berdasarkan rasio dan luas."""
    params = params or PipelineParams()
    height, width = edges.shape[:2]
    min_area = max(1.0, params.min_area_ratio * height * width)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    scored: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw < 5 or rh < 5:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not (params.aspect_min <= ratio <= params.aspect_max):
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if not 4 <= len(approx) <= 8:
            continue
        quad = cv2.boxPoints(rect).astype(np.float32)
        # Dekat rasio 2,8 dan luas besar lebih diutamakan.
        score = -abs(np.log(ratio / PREFERRED_RATIO)) + 0.2 * np.log1p(area / min_area)
        scored.append((float(score), quad))

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    for _, quad in scored:
        box = cv2.boundingRect(np.round(quad).astype(np.int32))
        if any(_box_iou(box, other) > 0.45 for other in boxes):
            continue
        chosen.append(quad)
        boxes.append(box)
        if len(chosen) >= params.max_candidates:
            break
    return chosen


def perspective_rectify(
    image: np.ndarray, quad: np.ndarray, size: tuple[int, int] = RECT_SIZE
) -> np.ndarray:
    """Tahap 5a: koreksi perspektif ke ukuran tetap."""
    src = _order_points(quad)
    dst = np.array(
        [[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_CUBIC)


def _ensure_light_background(binary: np.ndarray) -> np.ndarray:
    """OCR lebih stabil bila latar putih dan tulisan hitam."""
    return binary if float(binary.mean()) >= 127 else cv2.bitwise_not(binary)


def binarize_plate(rectified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tahap 5b: Otsu dan adaptif, keduanya dikembalikan untuk perbandingan."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return _ensure_light_background(otsu), _ensure_light_background(adaptive)


def segment_characters(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Tahap 6: kontur karakter, diurutkan dari kiri ke kanan."""
    height, width = binary.shape[:2]
    # Pinggiran plat yang gelap sering membentuk satu kontur besar yang
    # menutupi karakter. Buang margin sebelum mencari kontur huruf.
    mx, my = round(0.04 * width), round(0.10 * height)
    x0, y0 = min(mx, width // 4), min(my, height // 4)
    x1, y1 = max(x0 + 1, width - x0), max(y0 + 1, height - y0)
    work = binary[y0:y1, x0:x1]
    work_h, work_w = work.shape[:2]
    ink = cv2.bitwise_not(work)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bh < 0.18 * work_h or bh > 0.95 * work_h:
            continue
        if bw < max(3, 0.008 * work_w) or bw > 0.22 * work_w:
            continue
        if cv2.contourArea(contour) < 12:
            continue
        boxes.append((bx + x0, by + y0, bw, bh))

    boxes.sort(key=lambda box: box[0])
    filtered: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if filtered and box[0] < filtered[-1][0] + filtered[-1][2] * 0.65:
            last = filtered[-1]
            if box[2] * box[3] > last[2] * last[3]:
                filtered[-1] = box
            continue
        filtered.append(box)
    return filtered


def rebuild_line(binary: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Rakit kembali karakter hasil segmentasi menjadi satu baris untuk OCR."""
    if not boxes:
        return binary
    target_h = 64
    padding = 6
    patches = []
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(binary.shape[1], x + bw + padding), min(binary.shape[0], y + bh + padding)
        crop = binary[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = target_h / float(crop.shape[0])
        new_w = max(1, round(crop.shape[1] * scale))
        patches.append(cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC))
    if not patches:
        return binary

    gap = 18
    margin = 12
    total_w = sum(p.shape[1] for p in patches) + gap * (len(patches) - 1) + 2 * margin
    canvas = np.full((target_h + 2 * margin, total_w), 255, dtype=np.uint8)
    cursor = margin
    for patch in patches:
        canvas[margin : margin + target_h, cursor : cursor + patch.shape[1]] = patch
        cursor += patch.shape[1] + gap
    return canvas


def annotate_image(image: np.ndarray, plates: list[PlateCandidate]) -> np.ndarray:
    """Tahap 8: gambar kotak dan label hasil pada citra asli yang sudah diresize."""
    out = image.copy()
    coral_bgr = (128, 123, 244)
    for plate in plates:
        pts = np.round(plate.quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, coral_bgr, 3, cv2.LINE_AA)
        x, y, w, h = plate.bbox
        label = f"{plate.index}. {plate.text or '?'}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(0, y - th - baseline - 6)
        cv2.rectangle(out, (x, label_y), (x + tw + 12, label_y + th + baseline + 6), coral_bgr, -1)
        cv2.putText(
            out,
            label,
            (x + 6, label_y + th + baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def process_image(
    source: Any, params: PipelineParams | None = None
) -> PipelineResult:
    """Jalankan seluruh alur dan kembalikan citra tiap tahap + metrik waktu."""
    params = params or PipelineParams()
    started = perf_counter()
    timings: dict[str, float] = {}

    def accumulate(name: str, t0: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + (perf_counter() - t0) * 1000, 2)

    stages: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    t0 = perf_counter()
    bgr = load_image(source)
    accumulate("decode", t0)
    stages["original"] = bgr.copy()

    t0 = perf_counter()
    resized = resize_image(bgr, MAX_WIDTH)
    accumulate("resize", t0)
    stages["resize"] = resized.copy()

    t0 = perf_counter()
    gray, enhanced = enhance_image(resized, params.use_clahe)
    accumulate("enhance", t0)
    stages["grayscale"] = gray
    stages["enhanced"] = enhanced

    t0 = perf_counter()
    edges = detect_edges(enhanced, params.canny_low, params.canny_high)
    accumulate("edges", t0)
    stages["edges"] = edges

    t0 = perf_counter()
    quads = find_plate_candidates(edges, params)
    accumulate("contours", t0)
    contour_view = resized.copy()
    for quad in quads:
        cv2.polylines(
            contour_view,
            [np.round(quad).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (128, 123, 244),
            2,
            cv2.LINE_AA,
        )
    stages["contours"] = contour_view

    plates: list[PlateCandidate] = []
    ocr_error_shown = False
    for index, quad in enumerate(quads, start=1):
        t0 = perf_counter()
        rectified = perspective_rectify(resized, quad)
        accumulate("rectify", t0)
        stages[f"rectified_{index}"] = rectified.copy()

        t0 = perf_counter()
        otsu, adaptive = binarize_plate(rectified)
        accumulate("binarization", t0)
        stages[f"otsu_{index}"] = otsu.copy()
        stages[f"adaptive_{index}"] = adaptive.copy()

        variants = (
            [("otsu", otsu), ("adaptive", adaptive)]
            if params.binarization == "auto"
            else [(params.binarization, adaptive if params.binarization == "adaptive" else otsu)]
        )

        best_text = ""
        best_score = -1.0
        best_confidence = 0.0
        best_name = variants[0][0]
        best_binary = variants[0][1]
        best_boxes: list[tuple[int, int, int, int]] = []
        for name, binary in variants:
            t0 = perf_counter()
            boxes = segment_characters(binary)
            accumulate("segmentation", t0)
            line = rebuild_line(binary, boxes)
            # Hasil segmentasi dirakit menjadi satu baris, tetapi OCR juga
            # dijalankan pada plat utuh sebagai pembanding bila rakitan itu
            # justru menurunkan keyakinan Tesseract.
            ocr_inputs = [(line, "line"), (binary, "full")] if boxes else [(binary, "full")]
            for image_for_ocr, _source_kind in ocr_inputs:
                t0 = perf_counter()
                try:
                    output = read_plate(image_for_ocr)
                    score = output.confidence + 0.01 * len(output.normalized)
                    if score > best_score:
                        best_score = score
                        best_confidence = output.confidence
                        best_text = output.normalized
                        best_name = name
                        best_binary = binary
                        best_boxes = boxes
                except OcrUnavailableError as exc:
                    if not ocr_error_shown:
                        warnings.append(str(exc))
                        ocr_error_shown = True
                accumulate("ocr", t0)

        char_view = cv2.cvtColor(best_binary, cv2.COLOR_GRAY2BGR)
        for bx, by, bw, bh in best_boxes:
            cv2.rectangle(char_view, (bx, by), (bx + bw, by + bh), (128, 123, 244), 2)
        stages[f"characters_{index}"] = char_view

        score = float(
            -abs(np.log(max(1e-6, _aspect_of(quad)) / PREFERRED_RATIO))
        )
        plates.append(
            PlateCandidate(
                index=index,
                quad=quad,
                bbox=cv2.boundingRect(np.round(quad).astype(np.int32)),
                rectified=rectified,
                otsu=otsu,
                adaptive=adaptive,
                binary=best_binary,
                threshold_used=best_name,
                char_boxes=best_boxes,
                text=best_text,
                confidence=round(max(0.0, min(100.0, best_confidence)), 2),
                valid_pattern=validate_plate(best_text),
                score=score,
            )
        )

    if not quads:
        warnings.append(
            "Belum ada plat yang lolos filter. Coba turunkan ambang Canny, "
            "perbesar foto, atau ambil citra dengan plat lebih menghadap kamera."
        )

    stages["annotated"] = annotate_image(resized, plates)
    timings["total"] = round((perf_counter() - started) * 1000, 2)
    primary = plates[0] if plates else None
    return PipelineResult(
        stages=stages,
        annotated_image=stages["annotated"],
        plates=plates,
        text=primary.text if primary else "",
        texts=[plate.text for plate in plates],
        confidence=primary.confidence if primary else 0.0,
        timings_ms=timings,
        warnings=warnings,
    )


def _aspect_of(quad: np.ndarray) -> float:
    """Rasio sisi kotak hasil minAreaRect, dipakai sebagai skor ringan."""
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    widths = np.linalg.norm(points[[1, 3]] - points[[0, 2]], axis=1)
    widths = widths[widths > 0]
    if widths.size == 0:
        return PREFERRED_RATIO
    ratio = float(widths.max() / widths.min())
    return max(ratio, 1.0)
